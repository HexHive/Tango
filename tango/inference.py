from __future__ import annotations

from . import debug, info, warning, critical

from tango.core import (UniformStrategy, AbstractState, AbstractInput,
    BaseStateGraph, AbstractTracker, ValueProfiler, TimeElapsedProfiler,
    NumericalProfiler, LambdaProfiler, AbstractLoader, CountProfiler,
    EmptyInput, is_profiling_active, get_profiler, get_current_session,
    AbstractProfilerMeta as create_profiler)
from tango.cov import CoverageTracker, CoverageWebRenderer, CoverageWebDataLoader
from tango.reactive import ReactiveInputGenerator, HavocMutator
from tango.webui import create_svg
from tango.common import get_session_task_group
from tango.exceptions import StabilityException
from tango.havoc import havoc_handlers, RAND, MUT_HAVOC_STACK_POW2

from functools import partial, cached_property
from itertools import combinations
from pathlib import Path
from aiohttp import web
from typing import Optional, Sequence
from enum import Enum, auto
from nptyping import NDArray, Shape
from sklearn import tree
import numpy as np
import networkx as nx
import datetime
import asyncio
import json
import signal

utcnow = datetime.datetime.utcnow

__all__ = [
    'StateInferenceStrategy', 'StateInferenceTracker', 'InferenceWebRenderer'
]

NumericalValueProfiler = create_profiler('NumericalValueProfiler',
    (NumericalProfiler, ValueProfiler),
    {'value': ValueProfiler.value})
PercentProfiler = create_profiler('PercentProfiler',
    (NumericalValueProfiler,),
    {
        'value': property(lambda self: self._value * 100),
        '__str__': lambda self: f'{NumericalValueProfiler.__str__(self)}%'
    })

class InferenceMode(Enum):
    Discovery = auto()
    Diversification = auto()
    CrossPollination = auto()

class RecoveredStateGraph(BaseStateGraph):
    def __init__(self, **kwargs):
        self.graph_cls.__init__(self)
        LambdaProfiler('states')(lambda: len(self.nodes))

    def copy(self, **kwargs) -> RecoveredStateGraph:
        G = super(BaseStateGraph, self).copy(**kwargs)
        return G

class StateInferenceTracker(CoverageTracker,
        capture_paths=('strategy.disperse_heat',)):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return config['strategy'].get('type') == 'inference'

    def __init__(self, *, disperse_heat: bool=False, **kwargs):
        if disperse_heat:
            kwargs['track_heat'] = True
        super().__init__(**kwargs)
        LambdaProfiler('inferred_snapshots')(lambda: len(self.nodes))

        # properties
        self.mode = InferenceMode.Discovery
        self.capability_matrix = np.empty((0,0), dtype=object)
        self.collapsed_matrix = np.empty((0,0), dtype=object)
        self.nodes = {}
        self.node_arr = np.empty((0,), dtype=object)
        self.equivalence_map = {}
        self.equivalence_arr = np.empty((0,), dtype=int)
        self.equivalence_states = {}
        self.recovered_graph = RecoveredStateGraph()

    def reconstruct_graph(self, adj):
        dt = [('transition', object)]
        A = adj.astype(dt)

        G = nx.empty_graph(0, self.recovered_graph)
        n, m = A.shape

        G.add_nodes_from(range(n))
        # Get a list of all the entries in the array with nonzero entries. These
        # coordinates become edges in the graph. (convert to int from np.int64)
        edges = ((int(e[0]), int(e[1])) for e in zip(*A.nonzero()))
        fields = sorted(
            (offset, dtype, name) for name, (dtype, offset) in A.dtype.fields.items()
        )
        triples = (
            (
                u,
                v,
                {
                    name: val
                    for (_, dtype, name), val in zip(fields, A[u, v])
                },
            )
            for u, v in edges
        )
        G.add_edges_from(triples)

    @cached_property
    def unmapped_states(self):
        G = self.state_graph
        return G.nodes - self.nodes.keys()

    def update_transition(self, source: AbstractState,
            destination: AbstractState, input: AbstractInput, *,
            state_changed: bool, exc: Exception=None, **kwargs):
        if not exc and state_changed and not destination in self.state_graph:
            self.__dict__.pop('unmapped_states', None)
        rv = super().update_transition(source, destination, input,
            state_changed=state_changed, exc=exc, **kwargs)
        return rv

    def set_nodes(self, nodes: Sequence[AbstractState],
            eqv_map: Mapping[int, int]):
        self.__dict__.pop('unmapped_states', None)
        self.nodes.clear()
        self.nodes.update({ nodes[i]: i for i in range(len(nodes)) })
        self.node_arr = np.array(list(nodes), dtype=object)
        self.equivalence_map = {nodes[l]: s_idx for l, s_idx in eqv_map.items()}

        # the nodes array
        idx = np.arange(self.node_arr.size)
        self.equivalence_arr = np.vectorize(eqv_map.get, otypes=(int,))(idx)
        # get the index of the unique equivalence set each node maps to
        _, membership = np.unique(self.equivalence_arr, axis=0,
            return_inverse=True)
        # re-arrange eqv set indices so that members are grouped together
        order = np.argsort(membership)
        # get the boundaries of each eqv set in the ordered nodes array
        groups, split = np.unique(membership[order], return_index=True)
        # split the ordered nodes array into eqv groups
        eqvs = np.split(idx[order], split[1:])
        self.equivalence_states = {
            sidx: eqvs[i] for i, sidx in enumerate(groups)
        }

    def reindex(self, from_idx, to_idx):
        re_node_arr = self.node_arr[from_idx]
        re_node_fwd_map = { u: v for u, v in zip(from_idx, to_idx) }
        re_eqv_arr = self.equivalence_arr[from_idx]
        re_eqv_states = {
            sidx: to_idx[np.intersect1d(from_idx, eqv,
                assume_unique=True, return_indices=True)[1]]
            for sidx, eqv in self.equivalence_states.items()
        }
        return re_node_arr, re_node_fwd_map, re_eqv_arr, re_eqv_states

    def out_edges(self, state: AbstractState) -> Iterable[Transition]:
        if state in self.state_graph:
            try:
                node_idx = self.nodes[state]
                adj_idx, = self.capability_matrix[node_idx,:].nonzero()
                nbrs = self.node_arr[adj_idx]
                edges = self.capability_matrix[node_idx, adj_idx]
                return ((state, dst, inp)
                    for dst, edge in zip(nbrs, edges)
                        for inp in edge)
            except KeyError:
                return super().out_edges(state)
        else:
            return ()

    # we do not override in_edges, because by construction, there is always one
    # edge that reaches a snapshot (it is the edge we use to cross-test the
    # capabilitity of reaching this snapshot)

class StateInferenceStrategy(UniformStrategy,
        capture_components={'tracker', 'loader'},
        capture_paths=['strategy.inference_batch', 'strategy.disperse_heat',
            'strategy.extend_on_groups', 'strategy.recursive_collapse',
            'strategy.dt_predict', 'strategy.dt_extrapolate',
            'strategy.validate', 'strategy.broadcast_state_schedule',
            'strategy.dump_stats', 'fuzzer.work_dir']):
    def __init__(self, *, tracker: StateInferenceTracker,
            loader: AbstractLoader,
            work_dir: str,
            inference_batch: int=50,
            disperse_heat: bool=False,
            extend_on_groups: bool=False,
            recursive_collapse: bool=False,
            dt_predict: bool=False,
            dt_extrapolate: bool=False,
            validate: bool=False,
            broadcast_state_schedule: bool=False,
            dump_stats: bool=False, **kwargs):
        super().__init__(**kwargs)
        self._tracker = tracker
        self._loader = loader
        self._work_dir = Path(work_dir)
        self._inference_batch = inference_batch
        self._disperse_heat = disperse_heat
        self._extend_on_groups = extend_on_groups
        self._recursive_collapse = recursive_collapse
        self._dt_predict = dt_predict
        self._dt_extrapolate = dt_extrapolate
        self._validate = validate
        self._broadcast_state_schedule = broadcast_state_schedule
        self._auto_dump_stats = dump_stats
        if dt_predict:
            self._dt_clf = tree.DecisionTreeClassifier()
            self._dt_fit = False
        self.filter_sblgs = np.vectorize(lambda s: s in self.valid_targets)
        self._crosstest_timer = TimeElapsedProfiler('time_crosstest')
        self._crosstest_timer()

    async def initialize(self):
        await super().initialize()
        self._nprng = np.random.default_rng(
            seed=self._entropy.randint(0, 0xffffffff))
        self._profilers = (
            'time_elapsed', 'time_crosstest',
            'snapshots', 'states', 'inferred_snapshots',
            'total_savings', 'total_misses', 'total_hits',
            'eg_savings', 'eg_misses', 'eg_hits',
            'dt_savings', 'dt_misses', 'dt_hits',
            'dtex_savings', 'dtex_misses', 'dtex_hits',
            'snapshot_cov', 'total_cov')
        if is_profiling_active(*self._profilers):
            session = get_current_session()
            self._dump_path = \
                self._work_dir / f'crosstest_{session.id}.csv'
            self._dump_path.write_text(','.join(self._profilers) + '\n')
            loop = session.loop
            loop.add_signal_handler(
                signal.SIGUSR2, self._dump_profilers, session.id)
        else:
            self._auto_dump_stats = False
            del self._profilers

    def _dump_profilers(self, sid):
        with open(self._dump_path, 'at') as file:
            values = map(lambda p: get_profiler(p, None), self._profilers)
            file.write(','.join(str(v.value) if v else '' for v in values))
            file.write('\n')

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return config['strategy'].get('type') == 'inference'

    async def step(self, input: Optional[AbstractInput]=None):
        match self._tracker.mode:
            case InferenceMode.Discovery:
                if len(self._tracker.unmapped_states) >= self._inference_batch:
                    self._tracker.mode = InferenceMode.CrossPollination
                    self._step_interrupted = True
                else:
                    try:
                        await super().step(input)
                    finally:
                        if self._broadcast_state_schedule:
                            self._broadcast_state_energy(self._target)

            case InferenceMode.CrossPollination:
                self._crosstest_timer()
                cap, eqv_map, mask, nodes = await self.perform_cross_pollination()
                collapsed = cap[~mask,:][:,~mask]
                if self._recursive_collapse:
                    collapsed, eqv_map = self._collapse_until_stable(
                        collapsed, eqv_map)

                if self._dt_predict:
                    X = cap[mask,:][:,mask] != None
                    Y = np.vectorize(eqv_map.get, otypes=(int,))(
                        np.arange(X.shape[0]))
                    self._dt_clf.fit(X, Y)
                    self._dt_fit = True

                self._tracker.capability_matrix = cap[mask,:][:,mask]
                self._tracker.collapsed_matrix = collapsed
                self._tracker.set_nodes(nodes, eqv_map)
                self._tracker.reconstruct_graph(collapsed)
                self._tracker.mode = InferenceMode.Discovery
                self._crosstest_timer()
                if self._auto_dump_stats:
                    self._dump_profilers(get_current_session().id)

    @classmethod
    def _collapse_until_stable(cls, adj, eqv_map):
        last_adj = adj
        while True:
            adj, eqv_map, node_mask = cls._collapse_graph(adj, eqv_map,
                dual_axis=True)
            # mask out collapsed nodes
            adj = adj[~node_mask,:][:,~node_mask]
            if adj.shape == last_adj.shape:
                break
            last_adj = adj
        return adj, eqv_map

    @classmethod
    def _collapse_graph(cls, adj, orig_eqv=None, dual_axis=False):
        orig_eqv = orig_eqv or {i: i for i in range(adj.shape[0])}

        # construct equivalence sets
        stilde = cls._construct_equivalence_sets(adj, dual_axis=dual_axis)
        # remove strictly subsumed nodes
        adj, stilde, node_mask, sub_map = cls._eliminate_subsumed_nodes(adj,
            stilde, dual_axis=dual_axis)
        # collapse capability matrix where equivalence states exist
        ext_adj, eqv_map, node_mask = cls._collapse_adj_matrix(adj, stilde,
            node_mask, sub_map)

        # reconstruct eqv_map based on orig_eqv
        eqv_map = {i: eqv_map[s] for i, s in orig_eqv.items()}

        return ext_adj, eqv_map, node_mask

    @staticmethod
    def intersect1d_nosort(a, b, /):
        idx = np.indices((a.shape[0], b.shape[0]))
        equals = lambda i, j: a[i] == b[j]
        equals_ufn = np.frompyfunc(equals, 2, 1)
        match = equals_ufn(*idx)
        a_idx, b_idx = np.where(match)
        return a_idx, b_idx

    async def perform_cross_pollination(self):
        G = self._tracker.state_graph
        batch = self._entropy.sample(
            tuple(self._tracker.unmapped_states), k=self._inference_batch)
        node_set = set(G.nodes) & self._tracker.nodes.keys() | set(batch)
        nodes = np.array(list(node_set))

        to_idx, from_idx = self.intersect1d_nosort(nodes,
            self._tracker.node_arr)

        # get current adjacency matrix
        adj = G.adjacency_matrix(nodelist=nodes)

        # and current capability matrix
        cap = self._tracker.capability_matrix

        # mask out edges which have already been cross-tested
        edge_mask = adj != None
        mask_irow, mask_icol = np.meshgrid(to_idx, to_idx, indexing='ij')
        edge_mask[mask_irow, mask_icol] = True

        root_node = self._tracker.entry_state
        if root_node in node_set:
            root_idx, = np.where(nodes == root_node)
            # the root node has no incident edges
            edge_mask[:, root_idx] = True

        # get a new capability matrix, overlayed with new adjacencies
        cap = self._overlay_capabilities(cap, adj, from_idx, to_idx)

        # get a capability matrix extended with cross-pollination
        await self._extend_cap_matrix(cap, nodes, edge_mask, from_idx, to_idx)

        # collapse, single-axis
        cap, eqv_map, node_mask = self._collapse_graph(cap)

        assert len(eqv_map) == len(nodes)

        # translate indices back to nodes
        for i in np.where(node_mask)[0]:
            eqv_map.setdefault(i, -1)

        return cap, eqv_map, node_mask, nodes

    @staticmethod
    def _overlay_capabilities(cap, adj, from_idx, to_idx):
        from_irow, from_icol = np.meshgrid(from_idx, from_idx, indexing='ij')
        to_irow, to_icol = np.meshgrid(to_idx, to_idx, indexing='ij')
        adj[to_irow, to_icol] = cap[from_irow, from_icol]
        return adj

    def _spread_crosstests(self, cap, eqvs, eqv_mask, edge_mask):
        projected_done = 0
        projected_pending = 0
        for eqv in eqvs:
            if not eqv.size:
                continue
            eqv = eqv.astype(int)
            idx_bc, = np.where(eqv_mask)
            grid_bc = np.meshgrid(eqv, idx_bc, indexing='ij')
            mixed = np.logical_or.reduce(edge_mask[*grid_bc])

            # we count how many tests were skipped just by broadcasting existing
            # capabilities
            tmp_mask = edge_mask[*grid_bc]
            projected_pending += np.count_nonzero(~tmp_mask)

            edge_mask[*grid_bc] |= mixed
            projected_done += np.count_nonzero(tmp_mask != edge_mask[*grid_bc])

            untested, = np.where(~edge_mask[*grid_bc][0])
            untested_idx = np.arange(len(eqv) * len(untested))

            # for n untested nodes, we perform n tests in total, instead of
            # |eqv|*n; projected_done are then |eqv|*n-n
            projected_done += len(untested_idx) - len(untested)

            spread_idx = self._nprng.choice(untested_idx, untested.shape[0],
                replace=False)
            spread_untested = np.zeros((eqv.shape[0], untested.shape[0]),
                dtype=bool).flatten()
            spread_untested[spread_idx] = True
            spread_untested = spread_untested.reshape((eqv.shape[0], untested.shape[0]))
            spread = np.zeros((eqv.shape[0], idx_bc.shape[0]), dtype=bool)
            spread[:, untested] = spread_untested
            edge_mask[*grid_bc] |= ~spread

        return projected_done, projected_pending - projected_done

    @classmethod
    def _broadcast_capabilities(cls, cap, eqvs, eqv_mask, edge_mask):
        for eqv in eqvs:
            eqv = eqv.astype(int)
            idx_bc, = np.where(eqv_mask)
            grid_bc = np.meshgrid(eqv, idx_bc, indexing='ij')
            mixed = cls.combine_transitions(cap, where=tuple(grid_bc), axis=0)
            cap[*grid_bc] = mixed[idx_bc]
            edge_mask[*grid_bc] = True

    async def _extend_cap_matrix(self, cap, nodes, edge_mask, from_idx, to_idx):
        # we need the complete adj matrix to obtain inputs which may have been
        # batched out
        node_set = set(nodes)
        other_nodes = self._tracker.state_graph.nodes - node_set
        nodelist = np.concatenate((nodes, list(other_nodes)))
        orig_adj = self._tracker.state_graph.adjacency_matrix(nodelist=nodelist)

        def get_incident_input_vector(idx):
            inputs = cap[:,idx]
            if not np.any(inputs):
                inputs = orig_adj[:,idx]
                if not np.any(inputs):
                    return
            # we're assuming all inputs for a feature are equal
            return inputs[inputs != None][0]

        init_done = np.count_nonzero(edge_mask)
        init_pending = edge_mask.size - init_done
        if not init_pending:
            return

        def report_progress():
            current_done = np.count_nonzero(edge_mask) - init_done
            progress = PercentProfiler('progress')
            progress(current_done/init_pending)
            ValueProfiler('status')(f'cross_test ({progress})')

        _, node_fwd_map, _, eqv_states = \
            self._tracker.reindex(from_idx, to_idx)
        eqvs = np.array(list(eqv_states.values()), dtype=object)

        # set up a mask for processing new nodes only in cap
        eqv_mask = np.ones(cap.shape[0], dtype=bool)
        eqv_mask[to_idx] = False

        # we can only apply prediction optimizations when we've performed at
        # least one round of cross-testing
        should_predict = to_idx.size > 0
        if should_predict:
            count_skips = 0
            count_tests = 0

        should_validate = should_predict and self._validate
        if should_validate:
            shadow_mask = ~edge_mask
            valid_mask = edge_mask.copy()
            count_hits = 0
            count_misses = 0

        if should_predict and self._extend_on_groups:
            eg_count_skips = 0
            eg_count_tests = 0
            eg_count_hits = 0
            eg_count_misses = 0

            # spread the responsibility of tests across members of the set;
            # this only affects quadrant B
            if should_validate:
                init_mask = edge_mask.copy()
            skips, tests = self._spread_crosstests(cap, eqvs, eqv_mask, edge_mask)

            eg_count_skips += skips
            eg_count_tests += tests
            count_skips += skips

            # WARN we must not add `tests` here since the tests will be
            # performed and accounted for later; avoid double-counting
            # count_tests += tests

            if should_validate:
                # extend_on_groups just "skips" tests by setting the edge_mask
                # cells to True; thus, we add those cells to our shadow_mask
                shadow_mask |= init_mask ^ edge_mask

        if should_predict and self._dt_predict:
            dt_count_skips = 0
            dt_count_tests = 0
            dt_count_hits = 0
            dt_count_misses = 0

            if self._dt_extrapolate:
                dtex_count_skips = 0
                dtex_count_tests = 0
                dtex_count_hits = 0
                dtex_count_misses = 0

        uidx, = np.where(np.any(~edge_mask, axis=1))
        uidx_existing, idx, _ = np.intersect1d(
            uidx, to_idx, assume_unique=True, return_indices=True)
        remaining = np.ones_like(uidx, dtype=bool)
        remaining[idx] = False
        uidx_remaining = uidx[remaining]
        uidx = np.concatenate((uidx_existing, uidx_remaining))
        for eqv_idx in uidx:
            eqv_node = nodes[eqv_idx]

            # with DT, we are only concerned with quadrants C and D
            should_dt_predict = should_predict and eqv_idx not in to_idx

            vidx = None
            if self._dt_predict and should_dt_predict:
                # traverse the dt
                dt = self._dt_clf.tree_
                stack = [0]
                candidates = np.empty((0,), dtype=int)
                while stack:
                    cur = stack.pop()
                    if dt.children_left[cur] == dt.children_right[cur]:
                        # we've reached a leaf node
                        ((sidx,)) = dt.value[cur][0].nonzero()
                        candidates = np.append(candidates, sidx)
                        # at this point, egde_mask[:,to_idx] has been covered;
                        # and for grouped nodes, it is all True.
                        continue
                    # the DT is trained on the previous mapping
                    dst_idx = node_fwd_map.get(dt.feature[cur])
                    if not dst_idx:
                        # the snapshot no longer exists, we try both sides
                        stack.append(dt.children_left[cur])
                        stack.append(dt.children_right[cur])
                        continue

                    if edge_mask[eqv_idx, dst_idx]:
                        # can occur when multiple paths are traversed
                        # e.g., when the snapshot no longer exists as above
                        exists = cap[eqv_idx, dst_idx] is not None
                    else:
                        if should_validate:
                            # edge was actually tested, so we mark it as such
                            shadow_mask[eqv_idx, dst_idx] = False
                            valid_mask[eqv_idx, dst_idx] = True

                        dst_node = nodes[dst_idx]
                        inputv = get_incident_input_vector(dst_idx)
                        if (exists := await self._perform_one_cross_pollination(
                                eqv_node, dst_node, inputv)):
                            self._update_cap_matrix(cap, eqv_idx, dst_idx, inputv)

                        # mark edge as tested
                        edge_mask[eqv_idx, dst_idx] = True

                        dt_count_tests += 1
                        count_tests += 1

                        # report completion status
                        report_progress()

                    children = dt.children_left if exists else dt.children_right
                    stack.append(children[cur])

                # At this point, we may have multiple possible groupings
                candidates_eqv = np.vectorize(eqv_states.get,
                    otypes=(object,))(candidates)
                if (l := len(candidates_eqv)) > 1:
                    # we have more than one possible equivalence set;
                    # we train a small DT to differentiate the two, based on
                    # their updated cap matrices
                    critical(f"Multiple candidates ({l}) are not yet supported!")
                    candidates_eqv = candidates_eqv[(0,),...]

                # we choose a candidate
                candidate_eqv, = candidates_eqv
                # snapshot indices which the candidate can reproduce
                cap_eqv, = np.where(
                    np.logical_or.reduce(cap[candidate_eqv] != None, axis=0))

                # the set of edges which were not tested during prediction
                vidx_untested, = np.where(~edge_mask[eqv_idx,:])

                # the set of untested edges to all new snapshots (quadrant D)
                vidx_ungrouped = np.setdiff1d(vidx_untested, to_idx,
                    assume_unique=True)

                # the set of edges from the capability set of the candidate
                # (quadrant C)
                vidx_unpredicted = np.intersect1d(
                    vidx_untested,
                    np.intersect1d(cap_eqv, to_idx, assume_unique=True))
                dt_count_tests += vidx_unpredicted.size
                dt_count_skips += np.intersect1d(vidx_untested, to_idx,
                    assume_unique=True).size - vidx_unpredicted.size

                if self._dt_extrapolate:
                    # get the set of new snapshots that we did not test, that
                    # the candidate can reproduce (quadrant D)
                    # WARN This assumes that all of to_idx is processed before
                    # new snapshots; this is enforced by constructing uidx with
                    # to_idx first
                    init_size = vidx_ungrouped.size
                    vidx_ungrouped = np.intersect1d(vidx_ungrouped, cap_eqv,
                        assume_unique=True)
                    dtex_count_tests += vidx_ungrouped.size
                    dtex_count_skips += init_size - vidx_ungrouped.size

                # the final set of edges to be tested after prediction
                vidx_tobetested = np.union1d(
                    vidx_unpredicted,
                    vidx_ungrouped)

                vidx_tobeskipped = np.setdiff1d(vidx_untested, vidx_tobetested,
                    assume_unique=True)
                assert np.all(~edge_mask[eqv_idx, vidx_tobeskipped])
                edge_mask[eqv_idx, vidx_tobeskipped] = True
                count_skips += vidx_tobeskipped.size

                vidx = vidx_tobetested
            else:
                vidx, = np.where(~edge_mask[eqv_idx,:])

            if should_predict:
                count_tests += vidx.size

            if should_validate:
                valid_mask[eqv_idx, vidx] = True
                shadow_mask[eqv_idx, vidx] = False

            for dst_idx in vidx:
                dst_node = nodes[dst_idx]
                inputv = get_incident_input_vector(dst_idx)
                assert inputv

                if (exists := await self._perform_one_cross_pollination(
                        eqv_node, dst_node, inputv)):
                    self._update_cap_matrix(cap, eqv_idx, dst_idx, inputv)

                # mark edge as tested
                assert not edge_mask[eqv_idx, dst_idx]
                edge_mask[eqv_idx, dst_idx] = True

                # report completion status
                report_progress()

        if self._extend_on_groups:
            self._broadcast_capabilities(cap, eqvs, eqv_mask, edge_mask)

        if should_validate:
            assert np.all(edge_mask)
            v_uidx, v_vidx = shadow_mask.nonzero()
            for i, (eqv_idx, dst_idx) in enumerate(zip(v_uidx, v_vidx)):
                progress = PercentProfiler('progress')
                progress((i+1)/len(v_uidx))
                ValueProfiler('status')(f'validate ({progress})')
                eqv_node = nodes[eqv_idx]
                dst_node = nodes[dst_idx]

                inputv = get_incident_input_vector(dst_idx)
                assert inputv
                if not inputv:
                    count_hits += 1
                    valid_mask[eqv_idx, dst_idx] = True
                    continue
                exists = await self._perform_one_cross_pollination(
                    eqv_node, dst_node, inputv)
                is_valid = not (exists ^ (cap[eqv_idx, dst_idx] is not None))
                assert not valid_mask[eqv_idx, dst_idx]
                valid_mask[eqv_idx, dst_idx] = is_valid

                match (eqv_idx in to_idx, dst_idx in to_idx):
                    case (True, True):
                        raise RuntimeError("Impossible situation")
                    case (True, False):
                        if is_valid:
                            eg_count_hits += 1
                        else:
                            eg_count_misses += 1
                    case (False, True):
                        if is_valid:
                            dt_count_hits += 1
                        else:
                            dt_count_misses += 1
                    case (False, False):
                        if is_valid:
                            dtex_count_hits += 1
                        else:
                            dtex_count_misses += 1
                if is_valid:
                    count_hits += 1
                else:
                    count_misses += 1

            ValueProfiler('total_misses')(count_misses)
            ValueProfiler('total_hits')(count_hits)
            if self._extend_on_groups:
                ValueProfiler('eg_misses')(eg_count_misses)
                ValueProfiler('eg_hits')(eg_count_hits)
            if self._dt_predict:
                ValueProfiler('dt_misses')(dt_count_misses)
                ValueProfiler('dt_hits')(dt_count_hits)
                if self._dt_extrapolate:
                    ValueProfiler('dtex_misses')(dtex_count_misses)
                    ValueProfiler('dtex_hits')(dtex_count_hits)

        if should_predict:
            verify_skips = 0
            if self._dt_predict:
                verify_skips += dt_count_skips
                dt_processed = dt_count_tests + dt_count_skips
                if dt_processed:
                    PercentProfiler('dt_savings')(dt_count_skips / dt_processed)

                if self._dt_extrapolate:
                    verify_skips += dtex_count_skips
                    dtex_processed = dtex_count_tests + dtex_count_skips
                    if dtex_processed:
                        PercentProfiler('dtex_savings')(dtex_count_skips / dtex_processed)

            if self._extend_on_groups:
                verify_skips += eg_count_skips
                eg_processed = eg_count_tests + eg_count_skips
                if eg_processed:
                    PercentProfiler('eg_savings')(eg_count_skips / eg_processed)

            assert count_skips == verify_skips
            assert init_pending == count_skips + count_tests
            ratio = count_skips / init_pending
            PercentProfiler('total_savings')(ratio)

    async def _perform_one_cross_pollination(self, eqv_src: AbstractState,
            eqv_dst: AbstractState, inputs: Sequence[AbstractInput]):
        for inp in inputs:
            try:
                if not await self._perform_partial_cross_pollination(
                        eqv_src, eqv_dst, inp):
                    break
            except Exception:
                break
        else:
            # eqv_node matched all the responses of src to reach dst
            return True
        return False

    async def _perform_partial_cross_pollination(self, eqv_src: AbstractState,
            eqv_dst: AbstractState, input: AbstractInput):
        try:
            assert eqv_src == await self._explorer.reload_state(eqv_src,
                dryrun=True)
            await self._loader.apply_transition(
                (eqv_src, eqv_dst, input), eqv_src, update_cache=False)
            return True
        except StabilityException as ex:
            # TODO add new states to graph and match against them too?
            # current_state = ex.current_state
            return False
        except Exception:
            return False

    @staticmethod
    def _update_cap_matrix(cap, src_idx, dst_idx, inputs):
        row = cap[src_idx,:]
        if not (t := row[dst_idx]):
            row[dst_idx] = inputs
        else:
            warning("Appending to existing transition; this should not happen"
                " if the original graph is a tree.")
            row[dst_idx] = t | inputs

    @staticmethod
    def _construct_equivalence_sets(adj, dual_axis=False):
        stilde = set()
        is_nbr = adj != None

        if dual_axis:
            # we re-arrange is_nbr such that axis 0 returns both the
            # out-neighbors and in-neighbors
            is_nbr = np.array((is_nbr, is_nbr.T), dtype=is_nbr.dtype)
            is_nbr = np.swapaxes(is_nbr, 0, 1)

        # the nodes array
        idx = np.array(range(adj.shape[0]))
        # get the index of the unique equivalence set each node maps to
        _, membership = np.unique(is_nbr, axis=0, return_inverse=True)
        # re-arrange eqv set indices so that members are grouped together
        order = np.argsort(membership)
        # get the boundaries of each eqv set in the ordered nodes array
        _, split = np.unique(membership[order], return_index=True)
        # split the ordered nodes array into eqv groups
        eqvs = np.split(idx[order], split[1:])

        for eqv in eqvs:
            stilde.add(frozenset(eqv))
        return stilde

    @staticmethod
    def _map_indices(mask, bound):
        mask = np.where(mask)[0]
        mapping = {
            i: i - np.where(mask < i)[0].max(initial=-1) - 1 if not i in mask \
                else -1 \
            for i in range(bound)
        }
        return mapping

    @classmethod
    def _adjust_stilde(cls, stilde, mask):
        if isinstance(stilde, set):
            n_stilde = list(set(x) for x in stilde)
        else:
            n_stilde = stilde
        bound = max(y for x in n_stilde for y in x) + 1
        mapping = cls._map_indices(mask, bound)
        for eqv in n_stilde:
            n_eqv = {mapping[i] for i in eqv}
            eqv.clear()
            eqv |= n_eqv
        if isinstance(stilde, set):
            stilde.clear()
            stilde.update(frozenset(x) for x in n_stilde)

    @classmethod
    def _eliminate_subsumed_nodes(cls, adj, stilde, dual_axis=False):
        svee = set()
        is_nbr = adj != None
        nbrs_fn = lambda x: frozenset(np.where(x)[0])
        nbrs = np.apply_along_axis(nbrs_fn, 1, is_nbr)
        if dual_axis:
            nbrs_in = np.apply_along_axis(nbrs_fn, 0, is_nbr)
            nbrs = np.array((nbrs, nbrs_in))
            nbrs = np.swapaxes(nbrs, 0, 1)

        subsumes = lambda u, v: 1 if np.all(nbrs[u] > nbrs[v]) else \
            -1 if np.all(nbrs[u] < nbrs[v]) else 0
        subsumes_ufn = np.frompyfunc(subsumes, 2, 1)

        # create a list of (i,j) indices, without replacement
        # FIXME can this be done without allocating the index?
        gridx = list(zip(*combinations(range(adj.shape[0]), 2))) or [(),()]
        mask = np.zeros_like(adj, dtype=bool)
        mask[*gridx] = True

        sub = np.zeros_like(adj, dtype=int)
        # apply subsumption logic for all node pairs (i,j)
        sub[mask] = subsumes_ufn(*gridx).astype(int)
        sub[mask.T] = -sub.T[mask.T]
        sub = sub.clip(min=0).astype(bool)

        # get the unique sets of subsumed nodes
        subbeds, subbers = np.unique(sub, axis=0, return_index=True)
        subbeds = np.apply_along_axis(nbrs_fn, 1, subbeds)

        sub_map = {}
        mask = np.zeros(len(adj), dtype=bool)
        for us, v in zip(subbeds, subbers):
            if mask[v] or not us:
                continue
            svee.add(us)
            idx = []
            for u in us:
                if mask[u]:
                    continue
                idx.append(u)
                for x, y in sub_map.items():
                    if y == u:
                        idx.append(x)
                # idx.extend(np.where(sub[u] & ~mask))
                mask[u] = True
            for u in idx:
                sub_map[u] = v
            if idx:
                idx.append(v)
                out_edges = cls.combine_transitions(adj, rows=idx, axis=0)
                adj[v,:] = out_edges
                if dual_axis:
                    in_edges = cls.combine_transitions(adj, cols=idx, axis=1) \
                        .reshape(adj.shape[0])
                    adj[:,v] = in_edges

        stilde = [set(x) for x in stilde]
        for u in np.where(mask)[0]:
            for eqv in stilde:
                if sub_map[u] in eqv:
                    # we add the subsumed node to the subsumer's equivalence set
                    # even though it is not entirely correct, but it makes it
                    # easier to collapse the adj matrix later
                    eqv.add(u)
                else:
                    # discard u from all other eqv sets in S~
                    eqv.discard(u)

        # discard the empty set if it exists
        stilde = {frozenset(x) for x in stilde}
        stilde.discard(frozenset())
        return adj, stilde, mask, sub_map

    @classmethod
    def combine_transitions(cls, adj, /, *, where=None, **kwargs):
        axis_mask = np.zeros_like(adj, dtype=bool)
        if where is None:
            rows = kwargs.pop('rows', np.arange(adj.shape[0]))
            cols = kwargs.pop('cols', np.arange(adj.shape[1]))
            where = tuple(np.meshgrid(rows, cols, indexing='ij', copy=False))
        else:
            assert not {'rows', 'cols'} & kwargs.keys()
        axis_mask[where] = True
        combine = lambda t, r: t | r if t and r else t or r
        return np.frompyfunc(combine, 2, 1, identity=None)\
                 .reduce(adj, where=axis_mask, initial=frozenset(), **kwargs)

    @classmethod
    def _collapse_adj_matrix(cls, adj, stilde, mask, sub_map):
        eqv_mask = np.concatenate((mask, np.zeros(len(stilde), dtype=bool)))
        eqv_map = sub_map
        s_idx = 0
        adj = np.pad(adj, (0, len(stilde)), constant_values=None)
        for r, eqv in enumerate(stilde, start=mask.shape[0]):
            idx = np.array(list(eqv))
            adj[r,:] = cls.combine_transitions(adj, rows=idx, axis=0)
            adj[:,r] = cls.combine_transitions(adj, cols=idx, axis=1)
            adj[r,r] = cls.combine_transitions(adj, rows=idx, cols=idx, axis=None)

            eqv_mask[idx] = True
            for l in idx:
                eqv_map[l] = s_idx
            s_idx += 1

        return adj, eqv_map, eqv_mask

    def update_state(self, state: AbstractState, /, *args, exc: Exception=None,
            **kwargs):
        super().update_state(state, *args, exc=exc, **kwargs)
        if not self._broadcast_state_schedule or not state:
            return
        if not exc:
            self._broadcast_state_energy(state)

    def _broadcast_state_energy(self, state: AbstractState):
        try:
            j = self._tracker.nodes[state]
            sidx = self._tracker.equivalence_map[state]
            eqv = self._tracker.equivalence_states[sidx]
            for i in eqv:
                if i == j:
                    continue
                sblg = self._tracker.node_arr[i]
                self._energy_map[sblg] += 1
        except KeyError:
            pass

    async def reload_target(self) -> AbstractState:
        try:
            state = self.target_state
            j = self._tracker.nodes[state]
            sidx = self._tracker.equivalence_map[state]
            eqv = self._tracker.equivalence_states[sidx]
            if not self._disperse_heat:
                self._target = self._entropy.choice(self._tracker.node_arr[eqv])
            else:
                out_edges = ((
                    (src := self._tracker.node_arr[s]),
                    (dst for _, dst, _ in src.out_edges)
                    ) for s in eqv)
                tpairs = np.array([
                    (
                        src,
                        np.sum(np.sum(self._tracker._feature_heat[
                                np.asarray(dst._feature_mask).nonzero()])
                            for dst in dsts)
                    ) for (src, dsts) in out_edges]).transpose()
                tpairs = tpairs[:, self.filter_sblgs(tpairs[0])]
                warm_sblgs = tpairs[:, tpairs[1] != 0]
                warm_sblgs[1,:] = np.cumsum(
                    np.reciprocal(warm_sblgs[1,:].astype(float)))
                if warm_sblgs.size:
                    warm_sblgs[1,:] *= 0.5 / warm_sblgs[1,:].max()
                cold_sblgs = tpairs[:, tpairs[1] == 0]
                cold_sblgs[1,:] = np.linspace(
                    1, warm_sblgs[1,:].max(initial=0.),
                    num=cold_sblgs.shape[1], endpoint=False)[::-1]
                sblgs, cum_weights = np.column_stack((warm_sblgs, cold_sblgs))
                if sblgs.size:
                    self._target, = self._entropy.choices(
                        sblgs, cum_weights=cum_weights)
        except KeyError:
            pass
        return await super().reload_target()

class InferenceWebRenderer(CoverageWebRenderer):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return config['strategy'].get('type') == 'inference'

    def get_webui_factory(self):
        return partial(InferenceWebDataLoader, tracker=self._tracker,
                       **self._webui_kwargs)

class InferenceWebDataLoader(CoverageWebDataLoader):
    @property
    def fresh_graph(self):
        H = self._session._explorer.tracker.recovered_graph
        G = H.copy(fresh=True)
        # we also return a reference to the original in case attribute access is
        # needed
        return G, H

    async def track_node(self, *args, ret, **kwargs):
        state, new = ret
        state = self._session._explorer.tracker.equivalence_map.get(state)
        if state is None:
            return
        now = utcnow()
        if new:
            self._node_added[state] = now
        self._node_visited[state] = now
        await self.update_graph()

    async def track_edge(self, *args, ret, **kwargs):
        src, dst, new = ret
        src = self._session._explorer.tracker.equivalence_map.get(src)
        dst = self._session._explorer.tracker.equivalence_map.get(dst)
        if None in (src, dst):
            return
        now = utcnow()
        if new:
            self._edge_added[(src, dst)] = now
        self._edge_visited[(src, dst)] = now
        await self.update_graph()

    async def update_graph(self, *args, **kwargs):
        if not self._session._explorer.tracker.nodes:
            return

        # first we get a copy so that we can re-assign node and edge attributes
        G, H = self.fresh_graph

        node_sizes = {
            sidx: len(eqvs)
                for sidx, eqvs in
                    self._session._explorer.tracker.equivalence_states.items()
        }
        max_node_size = max(node_sizes.values())
        node_sizes = {
            sidx: size / max_node_size for sidx, size in node_sizes.items()
        }

        to_delete = []
        for node, data in G.nodes(data=True):
            if len(G.nodes) > 1 and len(G.in_edges(node)) == 0 \
                    and len(G.out_edges(node)) == 0:
                to_delete.append(node)
                continue
            age = (utcnow() - self._node_visited.get(node, self.NA_DATE)).total_seconds()
            coeff = self.fade_coeff(self._draw_update_fadeout, age)
            lerp = self.lerp_color(
                self.DEFAULT_NODE_COLOR,
                self.LAST_UPDATE_NODE_COLOR,
                coeff)
            fillcolor = self.format_color(*lerp)

            age = (utcnow() - self._node_added.get(node, self.NA_DATE)).total_seconds()
            coeff = self.fade_coeff(self._draw_update_fadeout, age)
            penwidth = self.lerp(
                self.DEFAULT_NODE_PEN_WIDTH,
                self.NEW_NODE_PEN_WIDTH,
                coeff)

            data.clear()
            data['fillcolor'] = fillcolor
            data['penwidth'] = penwidth
            if node == self._session._strategy.target_state:
                data['color'] = self.format_color(*self.TARGET_LINE_COLOR)
                data['penwidth'] = self.NEW_NODE_PEN_WIDTH
            else:
                data['color'] = self.format_color(*self.NODE_LINE_COLOR)
            data['width'] = 0.75 * node_sizes[node]
            data['height'] = 0.5 * node_sizes[node]
            data['fontsize'] = 14 * node_sizes[node]
            data['penwidth'] *= node_sizes[node]

        for node in to_delete:
            G.remove_node(node)

        for src, dst, data in G.edges(data=True):
            age = (utcnow() - self._edge_visited.get((src, dst), self.NA_DATE)).total_seconds()
            coeff = self.fade_coeff(self._draw_update_fadeout, age)
            lerp = self.lerp_color(
                self.DEFAULT_EDGE_COLOR,
                self.LAST_UPDATE_EDGE_COLOR,
                coeff)
            color = self.format_color(*lerp)

            age = (utcnow() - self._edge_added.get((src, dst), self.NA_DATE)).total_seconds()
            coeff = self.fade_coeff(self._draw_update_fadeout, age)
            penwidth = self.lerp(
                self.DEFAULT_EDGE_PEN_WIDTH,
                self.NEW_EDGE_PEN_WIDTH,
                coeff)

            state = dst
            data.clear()
            data['color'] = color
            data['penwidth'] = penwidth * node_sizes[state]
            if 'minimized' in H.edges[src, dst]:
                label = f"min={len(H.edges[src, dst]['minimized'].flatten())}"
                data['label'] = label

        G.graph["graph"] = {'rankdir': 'LR'}
        G.graph["node"] = {'style': 'filled'}
        P = nx.nx_pydot.to_pydot(G)
        svg = await create_svg(P)

        msg = json.dumps({
            'cmd': 'update_painting',
            'items': {
                'svg': svg
            }
        })
        await self._ws.send_str(msg)

# FIXME this component could benefit from composability
class InferenceInputGenerator(ReactiveInputGenerator,
        capture_components={'tracker'},
        capture_paths=('generator.broadcast_mutation_feedback',
            'strategy.disperse_heat')):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['strategy'].get('type') == 'inference'

    def __init__(self, *, tracker: StateInferenceTracker,
            broadcast_mutation_feedback: bool=False,
            disperse_heat: bool=False, **kwargs):
        super().__init__(**kwargs)
        self._tracker = tracker
        self._broadcast_mutation_feedback = broadcast_mutation_feedback
        self._disperse_heat = disperse_heat

    def select_candidate(self, state: AbstractState):
        if not self._disperse_heat:
            return super().select_candidate(state)
        else:
            in_edges = (inp for _,_,inp in state.in_edges)
            rv = tuple(zip(*(
                (inp, np.sum(self._tracker._feature_heat[
                        np.asarray(dst._feature_mask).nonzero()]))
                    for _,dst,inp in state.out_edges)))
            if rv:
                out_edges, temperatures = rv
            else:
                out_edges = temperatures = ()
            candidates = (*out_edges, *in_edges, *self.seeds, EmptyInput())
            weights = np.reciprocal(temperatures, dtype=float)
            out_weight = np.sum(weights)
            max_weight = out_weight / 0.7 or 1.0
            other_weights = np.linspace(out_weight, max_weight,
                num=len(candidates) - len(out_edges) + 1)[1:]
            weights = np.hstack((np.cumsum(weights), other_weights))
            return self._entropy.choices(candidates, cum_weights=weights)[0]

    def update_transition(self, source: AbstractState,
            destination: AbstractState, input: AbstractInput, /, *,
            state_changed: bool, orig_input: AbstractInput, exc: Exception=None,
            **kwargs):
        result = super().update_transition(source, destination, input,
                state_changed=state_changed, orig_input=orig_input, exc=exc,
                **kwargs)
        if not self._broadcast_mutation_feedback or not result:
            return
        reward, actions = result
        if not actions:
            return

        try:
            sidx = self._tracker.equivalence_map[source]
            eqv = self._tracker.equivalence_states[sidx]
            eqv = list(map(self._tracker.node_arr.__getitem__, eqv))
            eqv.remove(source)
            if not eqv:
                return result
            for sblg in eqv:
                if sblg not in self._state_model:
                    self._init_state_model(sblg, copy_from=source)
        except KeyError:
            return result

        for node in eqv:
            node_model = self._state_model[node]
            self._update_weights(node_model['actions'], actions, reward)

        if state_changed:
            # update feature counts
            fcount = self._count_features(source, destination)
            for node in eqv:
                node_model = self._state_model[node]
                node_model['features'] += fcount
                node_model['cum_features'] += fcount

        self._log_model(*eqv)

        return result
