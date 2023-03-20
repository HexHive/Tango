from __future__ import annotations

from . import debug, info, warning, critical

from tango.core import (UniformStrategy, AbstractState, AbstractInput,
    BaseStateGraph, AbstractTracker, ValueProfiler, TimeElapsedProfiler,
    ValueMeanProfiler, LambdaProfiler, AbstractLoader, CountProfiler,
    EmptyInput)
from tango.cov import CoverageTracker
from tango.reactive import ReactiveInputGenerator, ReactiveHavocMutator
from tango.webui import WebRenderer, WebDataLoader
from tango.common import get_session_task_group, ComponentOwner
from tango.exceptions import StabilityException
from tango.havoc import havoc_handlers, RAND, MUT_HAVOC_STACK_POW2

from functools import partial, cached_property
from aiohttp import web
from typing import Optional, Sequence
from enum import Enum, auto
from nptyping import NDArray, Shape
from sklearn import tree
import numpy as np
import networkx as nx
import datetime
import asyncio

__all__ = [
    'StateInferenceStrategy', 'StateInferenceTracker', 'InferenceWebRenderer'
]

class InferenceMode(Enum):
    Discovery = auto()
    Diversification = auto()
    CrossPollination = auto()

class RecoveredStateGraph(BaseStateGraph):
    def __init__(self, **kwargs):
        self.graph_cls.__init__(self)
        LambdaProfiler("states")(lambda: len(self.nodes))

    def copy(self, **kwargs) -> RecoveredStateGraph:
        G = super(BaseStateGraph, self).copy(**kwargs)
        return G

class StateInferenceTracker(CoverageTracker):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return config['strategy'].get('type') == 'inference'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # properties
        self.mode = InferenceMode.Discovery
        self.capability_matrix = np.empty((0,0), dtype=object)
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
        return super().update_transition(source, destination, input,
            state_changed=state_changed, exc=exc, **kwargs)

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

class StateInferenceStrategy(UniformStrategy,
        capture_components={'tracker', 'loader'},
        capture_paths=['strategy.inference_batch',
            'strategy.extend_on_groups', 'strategy.recursive_collapse',
            'strategy.dt_predict', 'strategy.dt_extrapolate',
            'strategy.dt_validate', 'strategy.broadcast_state_schedule']):
    def __init__(self, *, tracker: StateInferenceTracker,
            loader: AbstractLoader,
            inference_batch: Optional[str | int]=None,
            extend_on_groups: Optional[bool]=False,
            recursive_collapse: Optional[bool]=False,
            dt_predict: Optional[bool]=False,
            dt_extrapolate: Optional[bool]=False,
            dt_validate: Optional[bool]=False,
            broadcast_state_schedule: Optional[bool]=False, **kwargs):
        super().__init__(**kwargs)
        self._tracker = tracker
        self._loader = loader
        self._inference_batch = int(inference_batch or 50)
        self._extend_on_groups = extend_on_groups
        self._recursive_collapse = recursive_collapse
        self._dt_predict = dt_predict
        self._dt_extrapolate = dt_extrapolate
        self._dt_validate = dt_validate
        self._broadcast_state_schedule = broadcast_state_schedule
        if dt_predict:
            self._dt_clf = tree.DecisionTreeClassifier()
            self._dt_fit = False
        self._crosstest_timer = TimeElapsedProfiler('time_crosstest')
        self._crosstest_timer()

    async def initialize(self):
        await super().initialize()
        self._nprng = np.random.default_rng(
            seed=self._entropy.randint(0, 0xffffffff))

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
                    await super().step(input)
            case InferenceMode.CrossPollination:
                self._crosstest_timer()
                cap, eqv_map, mask, nodes = await self.perform_cross_pollination()
                self._tracker.capability_matrix = cap[mask,:][:,mask]
                self._tracker.set_nodes(nodes, eqv_map)

                collapsed = cap[~mask,:][:,~mask]
                if self._recursive_collapse:
                    collapsed, eqv_map = self._collapse_until_stable(
                        collapsed, eqv_map)
                self._tracker.reconstruct_graph(collapsed)
                self._tracker.mode = InferenceMode.Discovery
                self._crosstest_timer()

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
        nodes = np.array(G.nodes)

        to_idx, from_idx = self.intersect1d_nosort(nodes,
            self._tracker.node_arr)

        # get current adjacency matrix
        adj = G.adjacency_matrix
        # and current capability matrix
        cap = self._tracker.capability_matrix
        assert cap.shape == (len(from_idx), len(from_idx))

        # mask out edges which have already been cross-tested
        edge_mask = adj != None
        mask_irow, mask_icol = np.meshgrid(to_idx, to_idx, indexing='ij')
        edge_mask[mask_irow, mask_icol] = True
        # if there are nodes with no predecessors, we consider edges to those
        # nodes as masked (e.g., the root node)
        edge_mask[:,*np.where(np.all(adj == None, axis=0))] = True

        # get a new capability matrix, overlayed with new adjacencies
        cap = self._overlay_capabilities(cap, adj, from_idx, to_idx)

        # get a capability matrix extended with cross-pollination
        await self._extend_cap_matrix(cap, nodes, edge_mask, from_idx, to_idx)

        # collapse, single-axis
        cap, eqv_map, node_mask = self._collapse_graph(cap)

        if self._dt_predict:
            X = cap[node_mask,:][:,node_mask] != None
            Y = np.vectorize(eqv_map.get, otypes=(int,))(np.arange(X.shape[0]))
            self._dt_clf.fit(X, Y)
            self._dt_fit = True

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

        return projected_pending - projected_done

    @classmethod
    def _broadcast_capabilities(cls, cap, eqvs, eqv_mask, edge_mask):
        for eqv in eqvs:
            idx_bc = np.where(eqv_mask)[0]
            grid_bc = np.meshgrid(eqv, idx_bc, indexing='ij')
            mixed = cls.combine_transitions(cap[*grid_bc], axis=0)
            cap[*grid_bc] = mixed
            edge_mask[*grid_bc] = True

    async def _extend_cap_matrix(self, cap, nodes, edge_mask, from_idx, to_idx):
        init_done = np.count_nonzero(edge_mask)
        init_pending = edge_mask.size - init_done
        projected_pending = init_pending

        def report_progress():
            current_done = np.count_nonzero(edge_mask) - init_done
            if should_predict and not should_validate:
                current_done += dt_skips
            percent = f'{100*current_done/init_pending:.1f}%'
            ValueProfiler('status')(f'cross_test ({percent})')

        _, node_fwd_map, _, eqv_states = \
            self._tracker.reindex(from_idx, to_idx)
        eqvs = np.array(list(eqv_states.values()), dtype=object)

        # set up a mask for processing new nodes only in cap
        eqv_mask = np.ones(cap.shape[0], dtype=bool)
        eqv_mask[to_idx] = False

        should_predict = self._dt_predict and self._dt_fit
        if self._extend_on_groups:
            # spread the responsibility of tests across members of the set
            actual_done = self._spread_crosstests(cap, eqvs, eqv_mask, edge_mask)
            projected_pending -= actual_done

        if should_predict:
            dt_skips = 0
            dt_tests = 0

        uidx, = np.where(np.any(~edge_mask, axis=1))
        for eqv_idx in uidx:
            eqv_node = nodes[eqv_idx]

            should_predict_idx = should_predict and eqv_idx not in to_idx
            should_validate = should_predict_idx and self._dt_validate

            vidx = None
            if should_predict_idx:
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
                        dst_node = nodes[dst_idx]
                        inputs = cap[:,dst_idx]
                        # we're assuming all inputs for a feature are equal
                        inputs = inputs[inputs != None][0]
                        if (exists := await self._perform_one_cross_pollination(
                                eqv_node, dst_node, inputs)):
                            self._update_cap_matrix(cap, eqv_idx, dst_idx, inputs)

                        # mark edge as tested
                        edge_mask[eqv_idx, dst_idx] = True
                        projected_pending -= 1
                        dt_tests += 1

                        # report completion status
                        report_progress()

                    children = dt.children_left if exists else dt.children_right
                    stack.append(children[cur])

                vidx_all, = np.where(~edge_mask[eqv_idx,:])
                vidx_ungrouped = np.setdiff1d(vidx_all, to_idx,
                    assume_unique=True)
                projected_pending -= vidx_ungrouped.size
                dt_skips += np.setdiff1d(vidx_all, vidx_ungrouped,
                    assume_unique=True).size

                if self._dt_extrapolate:
                    candidates_eqv = np.vectorize(eqv_states.get,
                        otypes=(object,))(candidates)

                    if len(candidates_eqv) > 1:
                        # we have more than one possible equivalence set;
                        # we train a small DT to differentiate the two, based on
                        # their updated cap matrices
                        # FIXME this assumes that all of to_idx is processed
                        # before new snapshots; instead, this should be enforced
                        # by constructing uidx with to_idx first
                        critical("Multiple candidates are not yet supported!")
                        candidates_eqv = candidates_eqv[(0,),:]

                    assert len(candidates_eqv) == 1
                    eqv, = candidates_eqv
                    # snapshot indices which the group can reproduce
                    cap_eqv, = np.where(
                        np.logical_or.reduce(cap[eqv] != None, axis=0))
                    # get the set of new snapshots that we did not test, that
                    # the group can reproduce
                    vidx_extra = np.intersect1d(vidx_ungrouped, cap_eqv,
                        assume_unique=True)
                    # update the pending tests to include only extrapolated
                    savings = vidx_ungrouped.size - vidx_extra.size
                    vidx_ungrouped = vidx_extra
                    projected_pending += savings

                if not should_validate:
                    # we ignore cross-testing against grouped nodes under the
                    # assumption that DT predicted them correctly
                    vidx = vidx_ungrouped
            else:
                projected_pending -= np.count_nonzero(~edge_mask[eqv_idx,:])

            if vidx is None:
                # this happens in the following cases:
                # - dt_predict is False
                # - dt_predict is True but dt_fit is False
                # - eqv_idx is a previously-grouped node
                # - none of the above, and dt_validate is True
                vidx, = np.where(~edge_mask[eqv_idx,:])

            if should_validate:
                dt_misses = 0
                dt_hits = 0

            for dst_idx in vidx:
                dst_node = nodes[dst_idx]
                inputs = cap[:,dst_idx]
                if not np.any(inputs):
                    continue
                inputs = inputs[inputs != None][0]

                should_validate_idx = should_validate and dst_idx in to_idx
                exists = await self._perform_one_cross_pollination(
                        eqv_node, dst_node, inputs)

                if should_validate_idx:
                    if exists ^ (cap[eqv_idx, dst_idx] is not None):
                        dt_misses += 1
                    else:
                        dt_hits += 1
                if exists:
                    self._update_cap_matrix(cap, eqv_idx, dst_idx, inputs)

                # mark edge as tested
                edge_mask[eqv_idx, dst_idx] = True

                # report completion status
                report_progress()

            if should_validate:
                assert np.all(edge_mask[eqv_idx,])
                ValueMeanProfiler('dt_miss',
                    samples=self._inference_batch, decimal_digits=2)(dt_misses)
                ValueMeanProfiler('dt_hit',
                    samples=self._inference_batch, decimal_digits=2)(dt_hits)
        if should_predict:
            percent = 100 * dt_skips / (dt_tests + dt_skips)
            ValueMeanProfiler('dt_savings', samples=5)(percent)

        percent = 100 * projected_pending / init_pending
        ValueMeanProfiler('total_savings', samples=5)(percent)

        if self._extend_on_groups:
            self._broadcast_capabilities(cap, eqvs, eqv_mask, edge_mask)

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
                (eqv_src, eqv_dst, input), eqv_src, do_not_cache=True)
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
            row[dst_idx] = list(inputs)
        else:
            warning("Appending to existing transition; this should not happen"
                " if the original graph is a tree.")
            row[dst_idx] = list(t) + list(inputs)

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

        # create a grid of (i,j) indices
        gridx = np.indices(adj.shape)
        subsumes = lambda u, v: np.all(nbrs[u] > nbrs[v])
        subsumes_ufn = np.frompyfunc(subsumes, 2, 1, identity=False)
        # apply subsumption logic for all node pairs (i,j)
        sub = subsumes_ufn(*gridx).astype(bool)

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
                out_edges = cls.combine_transitions(adj[idx,:], axis=0)
                adj[v,:] = out_edges
                if dual_axis:
                    in_edges = cls.combine_transitions(adj[:,idx], axis=1) \
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

    @staticmethod
    def _combine_transitions(r, t):
        if not (t and r):
            return t or r
        combined = list(r)
        for elem in t:
            if not elem in combined:
                combined.append(elem)
        return combined

    @classmethod
    @property
    def combine_transitions(cls):
        return np.frompyfunc(cls._combine_transitions, 2, 1,
            identity=[]).reduce

    @classmethod
    def _collapse_adj_matrix(cls, adj, stilde, mask, sub_map):
        eqv_mask = np.concatenate((mask, np.zeros(len(stilde), dtype=bool)))
        eqv_map = sub_map
        s_idx = 0
        for eqv in stilde:
            idx = np.array(list(eqv))
            out_edges = cls.combine_transitions(adj[idx,:], axis=0)
            in_edges = cls.combine_transitions(adj[:,idx], axis=1).reshape(adj.shape[0], 1)
            self_edges = cls.combine_transitions(adj[idx,idx], axis=0, keepdims=True)
            self_edges = cls.combine_transitions(self_edges, keepdims=True)
            # s_idx = adj.shape[0]
            adj = np.vstack((adj, out_edges))
            adj = np.hstack((adj, np.vstack((in_edges, self_edges))))

            for l in idx:
                eqv_map[l] = s_idx
                eqv_mask[l] = True
            s_idx += 1

        return adj, eqv_map, eqv_mask

    def update_state(self, state: AbstractState, /, *args, exc: Exception=None,
            **kwargs):
        super().update_state(state, *args, exc=exc, **kwargs)
        if not self._broadcast_state_schedule or not state:
            return
        if not exc:
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

class InferenceWebRenderer(WebRenderer):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return config['strategy'].get('type') == 'inference'

    def get_webui_factory(self):
        return partial(InferenceWebDataLoader, **self._webui_kwargs)

class InferenceWebDataLoader(WebDataLoader):
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
        now = datetime.datetime.now()
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
        now = datetime.datetime.now()
        if new:
            self._edge_added[(src, dst)] = now
        self._edge_visited[(src, dst)] = now
        await self.update_graph()

# FIXME this component could benefit from composability
class InferenceInputGenerator(ReactiveInputGenerator,
        capture_components={'tracker'},
        capture_paths=('generator.broadcast_mutation_feedback',)):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['strategy'].get('type') == 'inference'

    def __init__(self, *, tracker: StateInferenceTracker,
            broadcast_mutation_feedback: bool=False, **kwargs):
        super().__init__(**kwargs)
        self._tracker = tracker
        self._broadcast_mutation_feedback = broadcast_mutation_feedback

    def select_candidate(self, state: AbstractState):
        try:
            sidx = self._tracker.equivalence_map[state]
            eqv = self._tracker.equivalence_states[sidx]
            eqv = list(map(self._tracker.node_arr.__getitem__, eqv))
            out_edges = (inp for s in eqv for _,_,inp in s.out_edges)
            in_edges = (inp for s in eqv for _,_,inp in s.in_edges)
            candidates = (*out_edges, *in_edges, *self.seeds, EmptyInput())
            return self._entropy.choice(candidates)
        except KeyError:
            return super().select_candidate(state)

    def generate(self, state: AbstractState) -> AbstractInput:
        if not self._broadcast_mutation_feedback:
            return super().generate(state)

        if (model := self._state_model.get(state)) is None:
            model = self._init_state_model(state)

        havoc_actions = self._entropy.choices(havoc_handlers,
            # we use probabilities as weights
            weights=map(lambda t: model['actions'][t][1], havoc_handlers),
            k=RAND(MUT_HAVOC_STACK_POW2, self._entropy) + 1
        )
        candidate = self.select_candidate(state)
        return ReactiveHavocMutator(candidate, havoc_actions, entropy=self._entropy)

    def update_transition(self, source: AbstractState,
            destination: AbstractState, input: AbstractInput, /, *,
            state_changed: bool, orig_input: AbstractInput, exc: Exception=None,
            **kwargs):
        result = super().update_transition(source, destination, input,
                state_changed=state_changed, orig_input=orig_input, exc=exc,
                **kwargs)
        if not self._broadcast_mutation_feedback or \
                not orig_input.decorated or not result:
            return
        reward, actions_taken, actions = result

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

        if actions_taken:
            for node in eqv:
                node_model = self._state_model[node]
                self._update_weights(node_model['actions'], actions,
                    reward)

        if state_changed:
            # update feature counts
            fcount = self._count_features(source, destination)
            for node in eqv:
                node_model = self._state_model[node]
                node_model['features'] += fcount
                node_model['cum_features'] += fcount

        if state_changed or actions_taken:
            self._log_model(*eqv)

        return result
