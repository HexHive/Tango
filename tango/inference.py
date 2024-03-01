from __future__ import annotations

from . import debug, info, warning, critical

from tango.core import (SeedableStrategy,
    AbstractState, AbstractInput, BaseStateGraph, AbstractTracker,
    ValueProfiler, TimeElapsedProfiler, NumericalProfiler, LambdaProfiler,
    AbstractLoader, CountProfiler, EmptyInput, is_profiling_active,
    get_profiler, get_current_session, AbstractProfilerMeta as create_profiler)
from tango.cov import (
    CoverageTracker, CoverageWebRenderer, CoverageWebDataLoader,
    FeatureSnapshot)
from tango.reactive import ReactiveInputGenerator, HavocMutator
from tango.webui import create_svg
from tango.common import get_session_task_group
from tango.exceptions import StabilityException
from tango.havoc import havoc_handlers, RAND, MUT_HAVOC_STACK_POW2

from functools import partial, cached_property, cache, wraps
from itertools import combinations
from collections import defaultdict
from pathlib import Path
from aiohttp import web
from types import SimpleNamespace as record
from typing import Optional, Sequence, TypeVar, Iterable, Any
from collections.abc import (Mapping as ABCMapping, Reversible as ABCReversible,
    Iterable as ABCIterable)
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

        # properties
        self.mode = InferenceMode.Discovery
        self.equivalence = InferenceMap()
        self.recovered_graph = RecoveredStateGraph()

        LambdaProfiler('inferred_snapshots')(
            lambda: len(self.equivalence.mapped_snapshots))
        LambdaProfiler('states')(
            lambda: len(self.equivalence.state_labels))

    @cache
    def _get_snapshot_features(self, snapshot: FeatureSnapshot) -> Sequence[int]:
        return np.asarray(snapshot._feature_mask).nonzero()

    def calculate_edge_temperature(self, sidx):
        assert self._track_heat
        return np.sum(
            np.sum(self._feature_heat[self._get_snapshot_features(memb)])
            for memb in self.equivalence.members(sidx))

    def reconstruct_graph(self):
        dt = [('transition', object)]
        A = self.equivalence.collapsed_matrix.astype(dt)

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
    def unmapped_snapshots(self):
        G = self.state_graph
        return G.nodes - self.equivalence.mapped_snapshots

    def update_transition(self, source: AbstractState,
            destination: AbstractState, input: AbstractInput, *,
            state_changed: bool, exc: Exception=None, **kwargs):
        if not exc and state_changed and not destination in self.state_graph:
            self.__dict__.pop('unmapped_snapshots', None)
        rv = super().update_transition(source, destination, input,
            state_changed=state_changed, exc=exc, **kwargs)
        return rv

    def update_equivalence(self, equivalence):
        self.__dict__.pop('unmapped_snapshots', None)
        self.equivalence = equivalence

    def out_edges(self, state: AbstractState) -> Iterable[Transition]:
        if state in self.state_graph:
            try:
                node_idx = self.equivalence.index(state)
                adj_idx, = self.equivalence.capability_matrix[node_idx,:].nonzero()
                nbrs = self.equivalence.snapshot(adj_idx)
                edges = self.equivalence.capability_matrix[node_idx, adj_idx]
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

class StateInferenceStrategy(SeedableStrategy,
        capture_components={'tracker', 'loader'},
        capture_paths=['strategy.inference_batch', 'strategy.disperse_heat',
            'strategy.extend_on_groups', 'strategy.recursive_collapse',
            'strategy.dt_predict', 'strategy.dt_extrapolate',
            'strategy.validate',
            'strategy.dynamic_inference', 'strategy.complete_inference',
            'strategy.learning_rate', 'strategy.learning_rounds',
            'strategy.min_energy', 'strategy.max_energy',
            'strategy.dump_stats', 'fuzzer.work_dir']):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return config['strategy'].get('type') == 'inference'

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
            dynamic_inference: bool=False,
            complete_inference: bool=False,
            learning_rate: float=0.1,
            learning_rounds: int=10,
            min_energy: int=100,
            max_energy: int=1000,
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
        self._dynamic_inference = dynamic_inference
        self._complete_inference = complete_inference
        self._learning_rate = learning_rate
        self._learning_rounds = learning_rounds
        self._auto_dump_stats = dump_stats
        if dt_predict:
            self._dt_clf = tree.DecisionTreeClassifier()
            self._dt_fit = False
        self.filter_sblgs = np.vectorize(lambda s: s in self.valid_targets)
        self._crosstest_timer = TimeElapsedProfiler('time_crosstest')
        self._crosstest_timer()

        self._target = None
        self._min_energy = min_energy
        self._max_energy = max_energy
        self._energies = self._tracker.equivalence.wrap(defaultdict(lambda:
            record(limit=self._min_energy, energy=self._min_energy)))

        LambdaProfiler('strat_target')(lambda: self._target)
        LambdaProfiler('strat_counter')(
            lambda: self._energies[self._target].energy)

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

    async def step(self, input: Optional[AbstractInput]=None):
        match self._tracker.mode:
            case InferenceMode.Discovery:
                if len(self._tracker.unmapped_snapshots) >= self._inference_batch:
                    self._tracker.mode = InferenceMode.CrossPollination
                    self._step_interrupted = True
                else:
                    try:
                        should_reset = False
                        while (rec := self._energies[self._target].energy) <= 0:
                            self._target = self.recalculate_target()
                            should_reset = True
                        if should_reset:
                            await self.reload_target()
                            self._step_interrupted = False
                        await super().step(input)
                    finally:
                        rec -= 1

            case InferenceMode.CrossPollination:
                while self._tracker.unmapped_snapshots:
                    equivalence = await self.perform_inference()
                    self._tracker.update_equivalence(equivalence)
                    if not self._complete_inference:
                        break
                self._tracker.reconstruct_graph()
                self.assign_state_energies()
                self._tracker.mode = InferenceMode.Discovery

    async def perform_inference(self, *,
            snapshots: Optional[frozenset[AbstractState]]=None,
            features: Optional[frozenset[AbstractState]]=None,
            equivalence: Optional[InferenceMap]=None) \
            -> InferenceMap:
        self._crosstest_timer()
        cap, sub, collapsed, eqv_map, snapshots, features = \
            await self.perform_cross_pollination(snapshots, features, equivalence)
        equivalence = InferenceMap(snapshots, features, eqv_map, collapsed, cap, sub)
        self._crosstest_timer()
        if self._auto_dump_stats:
            self._dump_profilers(get_current_session().id)
        return equivalence

    @classmethod
    def _collapse_until_stable(cls, adj, eqv_map):
        last_adj = adj
        while True:
            adj, eqv_map, row_mask, col_mask, sub = cls._collapse_graph(adj, eqv_map,
                dual_axis=True)
            # mask out collapsed nodes
            adj = adj[~row_mask,:][:,~col_mask]
            if adj.shape == last_adj.shape:
                break
            last_adj = adj
        return adj, eqv_map

    @classmethod
    def _collapse_graph(cls, adj, orig_eqv=None, dual_axis=False, **kwargs):
        orig_eqv = orig_eqv or {i: i for i in range(adj.shape[0])}

        # construct equivalence sets
        stilde = cls._construct_equivalence_sets(adj, dual_axis=dual_axis)
        # remove strictly subsumed nodes
        adj, stilde, sub_mask, sub, sub_map = cls._eliminate_subsumed_nodes(adj,
            stilde, dual_axis=dual_axis)
        # collapse capability matrix where equivalence states exist
        ext_adj, eqv_map, row_mask, col_mask = cls._collapse_adj_matrix(adj, stilde,
            sub_mask, sub_map, **kwargs)

        # reconstruct eqv_map based on orig_eqv
        eqv_map = {i: eqv_map[s] for i, s in orig_eqv.items()}

        return ext_adj, eqv_map, row_mask, col_mask, sub

    async def perform_cross_pollination(self, snapshots, features, equivalence):
        equivalence = equivalence or self._tracker.equivalence

        if snapshots is None:
            all_snapshots = frozenset(self._tracker.state_graph.nodes)
        else:
            all_snapshots = frozenset(snapshots)
        pop_snapshots = all_snapshots - equivalence.mapped_snapshots
        rem_snapshots = self._inference_batch or len(pop_snapshots)
        batch_snapshots = self._entropy.sample(
            tuple(pop_snapshots), k=min(rem_snapshots, len(pop_snapshots)))
        set_snapshots = all_snapshots & equivalence.mapped_snapshots | frozenset(batch_snapshots)
        snapshots = np.array(tuple(set_snapshots))

        if features is None:
            all_features = all_snapshots
            pop_features = pop_snapshots
            rem_features = rem_snapshots
            batch_features = batch_snapshots
            set_features = set_snapshots
            features = snapshots
        else:
            all_features = frozenset(features)
            pop_features = all_features - equivalence.feature_snapshots
            rem_features = self._inference_batch or len(pop_features)
            batch_features = self._entropy.sample(
                tuple(pop_features), k=min(rem_features, len(pop_features)))
            set_features = all_features & equivalence.feature_snapshots | frozenset(batch_features)
            features = np.array(tuple(set_features))

        symmetric = features is snapshots
        if not symmetric and set_features == set_snapshots:
            # could be that we end up with the same set anyway
            features = snapshots
            symmetric = True

        re_equivalence, \
                (s_from_idx, s_to_idx), (f_from_idx, f_to_idx), \
                s_fwd_map, f_fwd_map = \
            equivalence.reindex(
                snapshots, features, return_indices=True, return_fwd=True)

        if not pop_snapshots and not pop_features:
            # there's nothing to be done, so we'll just return previous results
            cap = re_equivalence.capability_matrix
            sub = re_equivalence.subsumption_matrix
            collapsed = re_equivalence.collapsed_matrix
            eqv_map = re_equivalence.eqv_map
            return cap, sub, collapsed, eqv_map, snapshots, features

        if self._dt_predict and self._dt_fit:
            # this is at least the second round of inference, so the DT had
            # already been trained; however, it had been trained on the old
            # order. We map the old indices to the new ones
            dt_features = self._dt_clf.tree_.feature
            dt_features[...] = np.vectorize(
                lambda u: v if (v := f_fwd_map.get(u)) is not None else -1,
                otypes=(int,))(dt_features)

        # get current adjacency matrix
        adj = self._tracker.state_graph.adjacency_matrix(
            rowlist=snapshots, collist=features)

        # and current capability matrix
        cap = re_equivalence.capability_matrix

        # mask out edges which have already been cross-tested
        edge_mask = adj != None
        mask_irow, mask_icol = np.meshgrid(s_to_idx, f_to_idx, indexing='ij')
        edge_mask[mask_irow, mask_icol] = True

        # get a new capability matrix, overlayed with new adjacencies
        adj[mask_irow, mask_icol] = cap[mask_irow, mask_icol]
        cap = adj

        root_snapshot = self._tracker.entry_state
        if root_snapshot in set_features:
            root_idx, = np.where(features == root_snapshot)
            # the root node has no incident edges
            edge_mask[:, root_idx] = True

        # get a capability matrix extended with cross-pollination
        await self._extend_cap_matrix(
            cap, snapshots, features, edge_mask, re_equivalence)

        # collapse, single-axis
        cap, eqv_map, row_mask, col_mask, sub = self._collapse_graph(cap,
            symmetric=symmetric)

        assert len(eqv_map) == len(snapshots)

        for i in np.where(row_mask)[0]:
            eqv_map.setdefault(i, -1)

        collapsed = cap[~row_mask,:][:,~col_mask]
        cap = cap[row_mask,:][:,col_mask]
        if symmetric and self._recursive_collapse:
            collapsed, eqv_map = self._collapse_until_stable(
                collapsed, eqv_map)

        if self._dt_predict:
            X = cap != None
            Y = np.vectorize(eqv_map.get, otypes=(int,))(
                np.arange(X.shape[0]))
            self._dt_clf.fit(X, Y)
            self._dt_fit = True

        return cap, sub, collapsed, eqv_map, snapshots, features

    def _spread_crosstests(self, cap, eqvs, feature_mask, edge_mask):
        projected_done = 0
        projected_pending = 0
        for eqv in eqvs:
            if not len(eqv):
                continue
            idx_bc, = np.where(feature_mask)
            grid_bc = np.meshgrid(list(eqv), idx_bc, indexing='ij')
            mixed = np.logical_or.reduce(edge_mask[*grid_bc])

            # we count how many tests were skipped just by broadcasting existing
            # capabilities
            tmp_mask = edge_mask[*grid_bc]
            projected_pending += np.count_nonzero(~tmp_mask)

            edge_mask[*grid_bc] |= mixed
            projected_done += np.count_nonzero(tmp_mask != edge_mask[*grid_bc])

            untested, = np.where(~mixed)
            untested_idx = np.arange(len(eqv) * len(untested))

            # for n untested nodes, we perform n tests in total, instead of
            # |eqv|*n; projected_done are then |eqv|*n-n
            projected_done += len(untested_idx) - len(untested)

            spread_idx = self._nprng.choice(untested_idx, untested.shape[0],
                replace=False)
            spread_untested = np.zeros((len(eqv), untested.shape[0]),
                dtype=bool).flatten()
            spread_untested[spread_idx] = True
            spread_untested = spread_untested.reshape((len(eqv), untested.shape[0]))
            spread = np.zeros((len(eqv), idx_bc.shape[0]), dtype=bool)
            spread[:, untested] = spread_untested
            edge_mask[*grid_bc] |= ~spread

        return projected_done, projected_pending - projected_done

    @classmethod
    def _broadcast_capabilities(cls, cap, eqvs, feature_mask, edge_mask):
        for eqv in eqvs:
            idx_bc, = np.where(feature_mask)
            grid_bc = np.meshgrid(list(eqv), idx_bc, indexing='ij')
            mixed = cls.combine_transitions(cap, where=tuple(grid_bc), axis=0)
            cap[*grid_bc] = mixed[idx_bc]
            edge_mask[*grid_bc] = True

    async def _extend_cap_matrix(self, cap, snapshots, features, edge_mask, equivalence):
        # we need the complete adj matrix to obtain inputs which may have been
        # batched out;
        # the order of rows does not matter for our use-case, because we only
        # access feature_adj with col indices matching `features`; specifically,
        # in `get_incident_input_vector` we use `idx` to access the matrix
        # column-wise to check the incident edges to snapshots in `features`.
        feature_adj = self._tracker.state_graph.adjacency_matrix(collist=features)

        def get_incident_input_vector(idx):
            inputs = feature_adj[:,idx]
            if not len(inputs):
                return

            # WARN we're assuming all inputs for a feature are equal
            b = inputs.astype(bool)
            res = b.argmax()
            if not b[res]:
                return
            return inputs[res]

        init_done = np.count_nonzero(edge_mask)
        init_pending = edge_mask.size - init_done
        if not init_pending:
            return

        def report_progress():
            if not is_profiling_active('progress'):
                return
            # current_done = np.count_nonzero(edge_mask) - init_done
            current_done = count_tests + count_skips
            progress = PercentProfiler('progress')
            progress(current_done/init_pending)
            ValueProfiler('status')(f'cross_test ({progress})')

        mapped_labels = np.asarray(list(equivalence.mapped_labels), dtype=int)
        feature_labels = np.asarray(list(equivalence.feature_labels), dtype=int)
        eqvs = np.array(list(equivalence.states_as_index.values()), dtype=object)

        # set up a mask for processing new features only in cap
        feature_mask = np.ones(cap.shape[1], dtype=bool)
        feature_mask[feature_labels] = False

        # we can only apply prediction optimizations when we've performed at
        # least one round of cross-testing
        should_predict = feature_labels.size > 0
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
            skips, tests = self._spread_crosstests(cap, eqvs, feature_mask, edge_mask)

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
            uidx, mapped_labels, assume_unique=True, return_indices=True)
        remaining = np.ones_like(uidx, dtype=bool)
        remaining[idx] = False
        uidx_remaining = uidx[remaining]
        uidx = np.concatenate((uidx_existing, uidx_remaining))
        for eqv_idx in uidx:
            eqv_node = snapshots[eqv_idx]

            try:
                # with DT, we are only concerned with quadrants C and D
                should_dt_predict = all((
                    self._dt_predict, should_predict,
                    eqv_idx not in mapped_labels)) and self._dt_fit

                vidx = None
                if should_dt_predict:
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
                            # at this point, egde_mask[:,mapped_labels] has been covered;
                            # and for grouped nodes, it is all True.
                            continue
                        dst_idx = dt.feature[cur]
                        if dst_idx == -1:
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

                            dst_node = features[dst_idx]
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
                    candidates_eqv = equivalence.members(candidates, as_index=True)
                    # We choose a candidate;
                    # equivalence.members may return None in case a group was not
                    # found in the mapping; we need to replace that by an empty
                    # iterable for use later in the code.
                    # A group may disappear from the mapping after re-indexing if
                    # all its members vanish (e.g. irreproducible)
                    combine = np.frompyfunc(lambda l, r:
                        (l or frozenset()) | (r or frozenset()), 2, 1,
                        identity=frozenset())
                    candidate_eqv = combine.reduce(candidates_eqv)

                    # snapshot indices which the candidate can reproduce
                    cap_eqv, = np.where(
                        np.logical_or.reduce(cap[list(candidate_eqv)] != None, axis=0))

                    # the set of edges which were not tested during prediction
                    vidx_untested, = np.where(~edge_mask[eqv_idx,:])

                    # the set of untested edges to all new snapshots (quadrant D)
                    vidx_ungrouped = np.setdiff1d(vidx_untested, feature_labels,
                        assume_unique=True)

                    # the set of edges from the capability set of the candidate
                    # (quadrant C)
                    vidx_unpredicted = np.intersect1d(
                        vidx_untested,
                        np.intersect1d(cap_eqv, feature_labels, assume_unique=True))
                    dt_count_tests += vidx_unpredicted.size
                    dt_count_skips += np.intersect1d(vidx_untested, feature_labels,
                        assume_unique=True).size - vidx_unpredicted.size

                    if self._dt_extrapolate:
                        # get the set of new snapshots that we did not test, that
                        # the candidate can reproduce (quadrant D)
                        # WARN This assumes that all of mapped_labels is processed before
                        # new snapshots; this is enforced by constructing uidx with
                        # mapped_labels first
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

                if should_validate:
                    valid_mask[eqv_idx, vidx] = True
                    shadow_mask[eqv_idx, vidx] = False

                for dst_idx in vidx:
                    dst_node = features[dst_idx]
                    inputv = get_incident_input_vector(dst_idx)
                    assert inputv

                    if (exists := await self._perform_one_cross_pollination(
                            eqv_node, dst_node, inputv)):
                        self._update_cap_matrix(cap, eqv_idx, dst_idx, inputv)

                    # mark edge as tested
                    assert not edge_mask[eqv_idx, dst_idx]
                    edge_mask[eqv_idx, dst_idx] = True

                    count_tests += 1

                    # report completion status
                    report_progress()
            except Exception as ex:
                # we failed, likely at loading a state; skip
                warning(f"Failed to cross-test {eqv_node} ({ex = })")
                skipped_pending = np.count_nonzero(~edge_mask[eqv_idx])
                count_tests += skipped_pending
                edge_mask[eqv_idx] = True
                continue

        if self._extend_on_groups:
            self._broadcast_capabilities(cap, eqvs, feature_mask, edge_mask)

        if should_validate:
            assert np.all(edge_mask)
            v_uidx, v_vidx = shadow_mask.nonzero()
            for i, (eqv_idx, dst_idx) in enumerate(zip(v_uidx, v_vidx)):
                progress = PercentProfiler('progress')
                progress((i+1)/len(v_uidx))
                ValueProfiler('status')(f'validate ({progress})')
                eqv_node = snapshots[eqv_idx]
                dst_node = features[dst_idx]

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

                match (eqv_idx in mapped_labels, dst_idx in feature_labels):
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
            eqv_dst: AbstractState, inputs: Sequence[AbstractInput]) \
            -> bool:
        success = True
        for inp in inputs:
            try:
                if not await self._perform_partial_cross_pollination(
                        eqv_src, eqv_dst, inp):
                    # some other exception occurred
                    success = False
            except StabilityException as ex:
                # we observed a state different that the expected destination
                success = False
                if self._dynamic_inference and ex.current_state != eqv_src:
                    if self._loader._restore or self._loader._driver._clear_cov:
                        # WARN we need to re-execute the input with an un-altered
                        # coverage map, because the original one was based on
                        # eqv_dst, which we failed to reach
                        assert eqv_src == await self._explorer.reload_state(
                            eqv_src)
                        try:
                            await self._loader._driver.execute_input(inp)
                        except Exception as ex:
                            # silently ignore any exceptions that arise
                            warning(f"Failed to re-apply input ({ex = })")
                            continue
                    # if this is an unseen state, the result will not be cached;
                    # we'll use this as an oracle instead of checking if the
                    # state is in the state graph
                    alt_dst = self._tracker.current_state
                    new_dst = self._tracker.update_state(eqv_src,
                        input=inp, peek_result=alt_dst)
                    assert new_dst == alt_dst
                    unseen = new_dst is not alt_dst
                    self._tracker.update_transition(eqv_src, new_dst, inp,
                        state_changed=eqv_src != new_dst, new_transition=unseen)
                    if unseen:
                        info(f"Discovered new state {new_dst} during inference")
                        # FIXME this callback interface is cursed
                        await self._explorer._state_update_cb(new_dst,
                            input=inp, orig_input=inp, breadcrumbs=new_dst)
                        # we call these to yield new files in the queue
                        await self._explorer._transition_update_cb(
                            eqv_src, new_dst, inp, orig_input=inp,
                            breadcrumbs=new_dst, state_changed=True,
                            new_transition=True)
                else:
                    # no need to try the rest if we're not interested in new
                    # responses
                    break
            except Exception:
                # we either failed to reload state, or failed to send data;
                # we'll raise this so that the state is skipped
                raise
        return success

    async def _perform_partial_cross_pollination(self, eqv_src: AbstractState,
            eqv_dst: AbstractState, input: AbstractInput) \
            -> Optional[AbstractState]:
        assert eqv_src == await self._explorer.reload_state(eqv_src)
        try:
            return await self._loader.apply_transition(
                (eqv_src, eqv_dst, input), eqv_src, update_cache=False)
        except StabilityException as ex:
            # we'll just inform the caller that we failed to reproduce the
            # transition; this would be used with dynamic_inference
            assert ex.expected_state is eqv_dst
            raise
        except Exception as ex:
            # otherwise, we ignore the exception and return a failure
            return None

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
            assert adj.shape[0] == adj.shape[1], ( \
                "Dual axis matching requires snapshots and features to be the"
                " same."
            )
            # we re-arrange is_nbr such that axis 0 returns both the
            # out-neighbors and in-neighbors
            is_nbr = np.array((is_nbr, is_nbr.T), dtype=is_nbr.dtype)
            is_nbr = np.swapaxes(is_nbr, 0, 1)

        # the snapshots array
        idx = np.array(range(adj.shape[0]))
        # get the index of the unique equivalence set each node maps to
        _, membership = np.unique(is_nbr, axis=0, return_inverse=True)
        # re-arrange eqv set indices so that members are grouped together
        order = np.argsort(membership)
        # get the boundaries of each eqv set in the ordered snapshots array
        _, split = np.unique(membership[order], return_index=True)
        # split the ordered snapshots array into eqv groups
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

        # FIXME can this be done without allocating the index?
        gridx = np.triu_indices(adj.shape[0], k=1)
        mask = np.zeros((adj.shape[0],)*2, dtype=bool)
        mask[*gridx] = True

        iax = slice_along_axis = lambda arr, index, axis: \
            arr[tuple(
                slice(None) if i != axis else index for i in range(arr.ndim))]
        subsumes = lambda u, v, ax: 0 if not \
            (x := (
                a := iax(is_nbr, u, ax)) ^ (b := iax(is_nbr, v, ax))).any() \
            else  1 if np.array_equal(x & a, x) \
            else -1 if np.array_equal(x & b, x) \
            else 0
        subsumes_ufn = np.frompyfunc(subsumes, 3, 1)

        # apply subsumption logic for all node pairs (i,j)
        all_sub = row_sub = subsumes_ufn(*gridx, 0, dtype=int)
        if dual_axis:
            assert adj.shape[0] == adj.shape[1], ( \
                "Dual axis matching requires snapshots and features to be the"
                " same."
            )
            col_sub = subsumes_ufn(*gridx, 1, dtype=int)
            overlap = row_sub == col_sub
            all_sub[~overlap] = 0

        sub = np.zeros_like(mask, dtype=int)
        sub[mask] = all_sub
        sub[mask.T] = -sub.T[mask.T]
        sub = sub.clip(min=0).astype(bool)

        # get the unique sets of subsumed nodes
        subbeds, subbers = np.unique(sub, axis=0, return_index=True)
        subbed_fn = lambda x: frozenset(np.where(x)[0])
        subbeds = np.apply_along_axis(subbed_fn, 1, subbeds)

        sub_map = {}
        sub_mask = np.zeros(adj.shape[0], dtype=bool)
        for us, v in zip(subbeds, subbers):
            if sub_mask[v] or not us:
                continue
            svee.add(us)
            idx = []
            for u in us:
                if sub_mask[u]:
                    continue
                idx.append(u)
                for x, y in sub_map.items():
                    if y == u:
                        idx.append(x)
                # idx.extend(np.where(sub[u] & ~sub_mask))
                sub_mask[u] = True
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
        for u in np.where(sub_mask)[0]:
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
        return adj, stilde, sub_mask, sub, sub_map

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
    def _collapse_adj_matrix(cls, adj, stilde, sub_mask, sub_map, *, symmetric=True):
        row_mask = np.concatenate((sub_mask, np.zeros(len(stilde), dtype=bool)))
        eqv_map = sub_map
        s_idx = 0
        pad = [(0, len(stilde)), (0, 0)]
        if symmetric:
            pad[1] = pad[0]
            col_mask = row_mask.copy()
        else:
            col_mask = np.ones(adj.shape[1], dtype=bool)

        rows, cols = adj.shape
        adj = np.pad(adj, pad, constant_values=None)
        for i, eqv in enumerate(stilde):
            row = i + rows
            idx = np.array(list(eqv))
            adj[row,:] = cls.combine_transitions(adj, rows=idx, axis=0)
            row_mask[idx] = True
            if symmetric:
                col = i + cols
                adj[:,col] = cls.combine_transitions(adj, cols=idx, axis=1)
                adj[row,col] = cls.combine_transitions(adj, rows=idx, cols=idx, axis=None)
                col_mask[idx] = True

            for l in idx:
                eqv_map[l] = s_idx
            s_idx += 1

        return adj, eqv_map, row_mask, col_mask

    def update_state(self, state: AbstractState, /, *args, exc: Exception=None,
            **kwargs):
        info(f"Updating states in {self}")
        super().update_state(state, *args, exc=exc, **kwargs)
        if state and not exc:
            if state is not self.target_state:
                rec = self._energies[state]
                rec.energy = max(0, rec.energy - 1)
                if not rec.energy:
                    # maybe the target always ends up here
                    self._target = self.recalculate_target()
        elif exc and self._target == state:
            self._target = self.recalculate_target()

    async def reload_target(self) -> AbstractState:
        try:
            snapshot = self.target_state
            siblings = self._tracker.equivalence.siblings(snapshot)
            self._target = self._entropy.choice(tuple(siblings))
        except KeyError:
            pass
        return await super().reload_target()

    def recalculate_target(self) -> AbstractState:
        filtered = self.valid_targets
        if not filtered:
            return None
        else:
            # FIXME a lot of work is repeated in indexing self._energies for
            # calculating siblings (see InferenceMap.wrap), but this is the
            # shortest solution for now...
            snapshot, rec = max(((s, self._energies[s]) for s in filtered),
                key=lambda t: t[1].energy)
            siblings = self._tracker.equivalence.siblings(snapshot)
            if rec.energy <= 0:
                # we've cycled through all states; reset energies
                self.reset_state_energies()
            return self._entropy.choice(tuple(siblings))

    def reset_state_energies(self):
        for rec in self._energies.values():
            rec.energy = min(rec.limit * 2, self._max_energy)

    def assign_state_energies(self):
        self._energies.clear()
        if self._disperse_heat:
            adj = self._tracker.recovered_graph.adjacency_matrix(
                weight=None, nonedge=0)
            heat_fn = np.vectorize(self._tracker.calculate_edge_temperature,
                otypes=(float,))
            edge_temperatures = heat_fn(np.arange(adj.shape[1]))
            state_temperatures = np.mean(adj * edge_temperatures, axis=1)
            for i in range(self._learning_rounds):
                # WARN this assumes that out_edges are selected uniformly
                # when searching for a candidate to mutate
                edge_temperatures = \
                    (1 - self._learning_rate) * edge_temperatures + \
                    self._learning_rate * state_temperatures
                state_temperatures = \
                    (1 - self._learning_rate) * state_temperatures + \
                    self._learning_rate * np.mean(adj * edge_temperatures,
                                                  axis=1)
            state_temperatures = state_temperatures.clip(min=np.nanmin(
                state_temperatures, initial=np.inf) or 0)
            state_energies = 1 / state_temperatures
            state_energies = (state_energies * self._max_energy / 2) / state_energies.max()
            state_energies = state_energies.clip(min=self._min_energy).astype(int)
            for (s,), e in np.ndenumerate(state_energies):
                # HACK due to how self._energies is wrapped and indexed, we need
                # to get one member snapshot, which will later get mapped to its
                # sibling set as a key into the energies dict;
                # Alternatively, we would iterate over all snapshots and assign
                # energies based on their set membership, but that may be even
                # more expensive, and will involve duplicate assignments.
                one_member = next(iter(self._tracker.equivalence.members(s)))
                rec = self._energies[one_member]
                rec.limit = rec.energy = e
        else:
            # self._energies is a defaultdict which initializes the energy
            # record with self._min_energy for any entry, so we do not need to
            # do it manually
            pass

    @property
    def target_state(self) -> AbstractState:
        return self._target

    @property
    def target(self) -> AbstractState:
        return self._target

    def update_transition(self, *args, **kwargs):
        pass

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
        snapshot, new = ret
        try:
            state = self._session._explorer.tracker.equivalence.state(snapshot)
        except KeyError:
            return
        now = utcnow()
        if new:
            self._node_added[state] = now
        self._node_visited[state] = now
        await self.update_graph()

    async def track_edge(self, *args, ret, **kwargs):
        src, dst, new = ret
        try:
            src = self._session._explorer.tracker.equivalence.state(src)
        except KeyError:
            src = None
        try:
            dst = self._session._explorer.tracker.equivalence.state(dst)
        except KeyError:
            dst = None
        now = utcnow()
        if src and dst:
            if new:
                self._edge_added[(src, dst)] = now
            self._edge_visited[(src, dst)] = now
        for n in (src, dst):
            if n:
                self._node_visited[n] = now
        await self.update_graph()

    async def update_graph(self, *args, **kwargs):
        # first we get a copy so that we can re-assign node and edge attributes
        G, H = self.fresh_graph

        if not G.nodes:
            return

        node_sizes = {
            sidx: max((rec := \
                self._session._strategy._energies[next(iter(eqvs))]).energy, 1)
            for sidx, eqvs in
                self._session._explorer.tracker.equivalence.states.items()
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
            siblings = set(self._tracker.equivalence.siblings(source))
            siblings.remove(source)
            if not siblings:
                return result
            for sblg in siblings:
                if sblg not in self._state_model:
                    self._init_state_model(sblg, copy_from=source)
        except KeyError:
            return result

        for node in siblings:
            node_model = self._state_model[node]
            self._update_weights(node_model['actions'], actions, reward)

        if state_changed:
            # update feature counts
            fcount = self._count_features(source, destination)
            for node in siblings:
                node_model = self._state_model[node]
                node_model['features'] += fcount
                node_model['cum_features'] += fcount

        self._log_model(*siblings)

        return result

T = TypeVar('T')
U = TypeVar('U')
SliceableIndex = T | Iterable[T]
class ReversibleMap(ABCMapping, ABCReversible):
    def __init__(self, *args,
            biject: bool=True, ktype=None, vtype=None, **kwargs):
        if kwargs:
            self._dict = dict(**kwargs)
        elif not args:
            self._dict = dict()
        elif isinstance(args[0], ABCMapping):
            self._dict = dict(*args)
        else:
            if len(args) == 1 and not isinstance(args := args[0], ABCIterable):
                raise TypeError(f"{args} is not a valid key-value iterable")
            self._dict = dict()
            for k, v in args:
                if (s := self._dict.get(k)):
                    s.add(v)
                    biject = False
                elif isinstance(k, frozenset):
                    for e in k:
                        vs = self._dict.setdefault(e, set())
                        vs.add(v)
                else:
                    self._dict[k] = {v}
            if biject:
                for k, v in self._dict.items():
                    self._dict[k] = v.pop()
                    assert not v, "biject==True, but the set is not empty"
            else:
                vtype = vtype and object
                for k, v in self._dict.items():
                    self._dict[k] = frozenset(v)
        self.ktype = ktype
        self.vtype = vtype

    def reverse(self, **kwargs):
        rev = self.__class__((reversed(t) for t in self._dict.items()),
            ktype=self.vtype, vtype=self.ktype, **kwargs)
        # HACK force cache this reversed version
        self.__dict__['_reversed'] = rev
        return rev

    @cached_property
    def as_dict(self) -> dict:
        return self._dict

    @cached_property
    def _reversed(self) -> ReversibleMap:
        return self.reverse()

    @cached_property
    def _vectorized(self):
        kw = {}
        if self.vtype:
            kw['otypes'] = (self.vtype,)
        return np.vectorize(self._dict.get, **kw)

    def __reversed__(self):
        return self._reversed

    def __getitem__(self, key: SliceableIndex[T]):
        if isinstance(key, ABCIterable):
            return self._vectorized(key)
        else:
            return self._dict[key]

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

class InferenceMap:
    def __init__(self,
            snapshots: Sequence[AbstractState]=(),
            features: Sequence[AbstractState]=(),
            eqv_map: Mapping[int, int]={},
            collapsed: NDArray[Shape["States, Features"]]=np.empty((0,0), dtype=object),
            cap: NDArray[Shape["Snapshots, Features"]]=np.empty((0,0), dtype=object),
            sub: NDArray[Shape["Snapshots, Snapshots"]]=np.empty((0,0), dtype=int)):
        self._i2s = ReversibleMap(enumerate(snapshots), ktype=int, vtype=object)
        self._f2s = ReversibleMap(enumerate(features), ktype=int, vtype=object)
        self._i2g = ReversibleMap(eqv_map, ktype=int, vtype=int)
        # HACK force cache the reverse, but with sets as values
        self._i2g.reverse(biject=False)

        cap_map = {i: frozenset(*np.where(cap[i,:])) for i in self.mapped_labels}
        self._i2f = ReversibleMap(cap_map, ktype=int, vtype=object)
        # HACK force cache the reverse, but with sets as values
        self._i2f.reverse(biject=False)

        self.capability_matrix = cap
        self.subsumption_matrix = sub
        self.collapsed_matrix = collapsed

        if sub.size:
            subsumers = {
                i: frozenset(*col.nonzero())
                for i, col in enumerate(
                    np.nditer(sub, flags=('external_loop',), order='F'))
            }
        else:
            subsumers = {}
        self._sub = ReversibleMap(subsumers, ktype=int, vtype=int)
        # HACK force cache the reverse, but with sets as values
        self._sub.reverse(biject=False)

    def siblings(self, snapshot: SliceableIndex[AbstractState | int], **kwargs) \
            -> frozenset[AbstractState | int]:
        try:
            return self.members(self.state(snapshot), **kwargs)
        except KeyError:
            if not kwargs.get('as_index', False):
                return frozenset((snapshot,))
            raise

    def state(self, snapshot: SliceableIndex[AbstractState | int]) -> int:
        resolve = lambda s: \
            self._i2g[self.index(s) if isinstance(s, AbstractState)
                      else s if isinstance(s, int) else None]
        if isinstance(snapshot, ABCIterable):
            resolve = np.vectorize(resolve, otypes=(int,))
        return resolve(snapshot)

    def index(self, snapshot: SliceableIndex[AbstractState]) -> int:
        return reversed(self._i2s)[snapshot]

    def snapshot(self, index: SliceableIndex[int]) -> AbstractState:
        return self._i2s[index]

    def members(self, sidx: SliceableIndex[int], as_index: bool=False) \
            -> frozenset[AbstractState | int]:
        idx = reversed(self._i2g)[sidx]
        if as_index:
            return idx
        else:
            freeze = lambda idx: frozenset(self.snapshot(i) for i in idx)
            if isinstance(sidx, ABCIterable):
                freeze = np.vectorize(freeze, otypes=(object,))
            return freeze(idx)

    def subsumers(self, snapshot: SliceableIndex[AbstractState | int], \
            as_index: bool=False) -> frozenset[AbstractState | int]:
        resolve = lambda s: \
            self._sub[self.index(s) if isinstance(s, AbstractState)
                      else s if isinstance(s, int) else None]
        if isinstance(snapshot, ABCIterable):
            resolve = np.vectorize(resolve, otypes=(object,))
        idx = resolve(snapshot)
        if as_index:
            return idx
        else:
            freeze = lambda idx: frozenset(self.snapshot(i) for i in idx)
            if isinstance(snapshot, ABCIterable):
                freeze = np.vectorize(freeze, otypes=(object,))
            return freeze(idx)

    def subsumees(self, snapshot: SliceableIndex[AbstractState | int], \
            as_index: bool=False) -> frozenset[AbstractState | int]:
        resolve = lambda s: \
            reversed(self._sub).get(self.index(s) if isinstance(s, AbstractState)
                      else s if isinstance(s, int) else None, frozenset())
        if isinstance(snapshot, ABCIterable):
            resolve = np.vectorize(resolve, otypes=(object,))
        idx = resolve(snapshot)
        if as_index:
            return idx
        else:
            freeze = lambda idx: frozenset(self.snapshot(i) for i in idx)
            if isinstance(snapshot, ABCIterable):
                freeze = np.vectorize(freeze, otypes=(object,))
            return freeze(idx)

    @property
    def mapped_snapshots(self) -> Iterable[AbstractState]:
        return reversed(self._i2s).keys()

    @property
    def mapped_labels(self) -> Iterable[int]:
        return self._i2g.keys()

    @property
    def feature_snapshots(self) -> Iterable[AbstractState]:
        return reversed(self._f2s).keys()

    @property
    def feature_labels(self) -> Iterable[int]:
        return reversed(self._i2f).keys()

    @property
    def state_labels(self):
        return reversed(self._i2g).keys()

    @property
    def states(self) -> Mapping[int, frozenset[AbstractState]]:
        return dict((s, self.members(s)) for s in self.state_labels)

    @property
    def states_as_index(self) -> Mapping[int, frozenset[int]]:
        return dict((s, self.members(s, as_index=True)) for s in self.state_labels)

    @property
    def eqv_map(self) -> Mapping[int, int]:
        return self._i2g.as_dict

    @property
    def cap_map(self) -> Mapping[int, int]:
        return self._i2f.as_dict

    def wrap(self, mapping: ABCMapping[T, Any], key: Callable[[U], T]=None):
        key = key or self.siblings
        class wrapper:
            @wraps(mapping.__getitem__)
            def __getitem__(self, k):
                if not k in mapping:
                    k = key(k)
                return mapping.__getitem__(k)
            def __getattr__(self, name):
                return getattr(mapping, name)
        return wrapper()

    @staticmethod
    def intersect1d_nosort(a, b, /):
        idx = np.indices((a.shape[0], b.shape[0]))
        equals = lambda i, j: a[i] == b[j]
        equals_ufn = np.frompyfunc(equals, 2, 1)
        match = equals_ufn(*idx)
        a_idx, b_idx = np.where(match)
        return a_idx, b_idx

    def reindex(self,
            new_snapshots: Sequence[AbstractState],
            new_features: Sequence[AbstractState], *,
            return_indices: bool=False,
            return_fwd: bool=False, return_rev: bool=False):
        old_snapshots = np.asarray(list(self.mapped_snapshots))
        old_features = np.asarray(list(self.feature_snapshots))
        s_to_idx, s_from_idx = self.intersect1d_nosort(
            new_snapshots, old_snapshots)
        f_to_idx, f_from_idx = self.intersect1d_nosort(
            new_features, old_features)

        re_eqv_map = {a: self._i2g[b] for a, b in zip(s_to_idx, s_from_idx)}

        from_irow, from_icol = np.meshgrid(s_from_idx, f_from_idx, indexing='ij')
        to_irow, to_icol = np.meshgrid(s_to_idx, f_to_idx, indexing='ij')
        re_cap = np.empty((len(new_snapshots), len(new_features)), dtype=object)
        re_cap[to_irow, to_icol] = self.capability_matrix[from_irow, from_icol]

        # WARN here we use s_*_idx for both dimensions, since the subsumption
        # matrix is always snapshot-oriented (no features are included)
        from_irow, from_icol = np.meshgrid(s_from_idx, s_from_idx, indexing='ij')
        to_irow, to_icol = np.meshgrid(s_to_idx, s_to_idx, indexing='ij')
        re_sub = np.zeros((len(new_snapshots),)*2, dtype=int)
        re_sub[to_irow, to_icol] = self.subsumption_matrix[from_irow, from_icol]

        # WARN collapsed_matrix does not need to be re-indexed, because states
        # do not change
        re_collapsed = self.collapsed_matrix

        re_equivalence = self.__class__(
            new_snapshots, new_features, re_eqv_map, re_collapsed, re_cap, re_sub)

        rv = re_equivalence,
        if return_indices:
            rv = rv + ((s_from_idx, s_to_idx), (f_from_idx, f_to_idx))
        if return_fwd:
            s_fwd_map = dict(zip(s_from_idx, s_to_idx))
            f_fwd_map = dict(zip(f_from_idx, f_to_idx))
            rv = rv + (s_fwd_map, f_fwd_map)
        if return_rev:
            s_rev_map = dict(zip(s_to_idx, s_from_idx))
            f_rev_map = dict(zip(f_to_idx, f_from_idx))
            rv = rv + (s_rev_map, f_rev_map)
        if len(rv) > 1:
            return rv
        else:
            return re_equivalence
