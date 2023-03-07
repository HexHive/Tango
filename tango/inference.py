from __future__ import annotations

from . import debug, info, warning

from tango.core import (UniformStrategy, AbstractState, AbstractInput,
    BaseStateGraph, AbstractStateTracker, ValueProfiler, TimeElapsedProfiler)
from tango.cov import CoverageStateTracker
from tango.webui import WebRenderer, WebDataLoader
from tango.common import get_session_task_group

from aiohttp import web
from typing import Optional
from enum import Enum, auto
from nptyping import NDArray, Shape
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

    def copy(self, **kwargs) -> RecoveredStateGraph:
        G = super(BaseStateGraph, self).copy(**kwargs)
        return G

class InferenceTracker(AbstractStateTracker):
    def __init__(self, **kwargs):
        self._graph = RecoveredStateGraph()

    def reconstruct_graph(self, adj):
        dt = [('transition', object)]
        A = adj.astype(dt)

        G = nx.empty_graph(0, self._graph)
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

    @classmethod
    def match_config(cls, config: dict) -> bool:
        # should never be instantiated alone
        return False

    @property
    def state_graph(self) -> RecoveredStateGraph:
        return self._graph

    ## Empty methods that will never be accessed

    def update_state(self, state: AbstractState, *,
            input: AbstractInput, exc: Exception=None, **kwargs) -> Any:
        raise NotImplementedError
    def update_transition(self, source: AbstractState,
            destination: AbstractState, input: AbstractInput, *,
            state_changed: bool, exc: Exception=None, **kwargs) -> Any:
        raise NotImplementedError

    @property
    def entry_state(self) -> AbstractState:
        raise NotImplementedError

    @property
    def current_state(self) -> AbstractState:
        raise NotImplementedError

    def peek(self, default_source: AbstractState, expected_destination: AbstractState) -> AbstractState:
        raise NotImplementedError

    def reset_state(self, state: AbstractState):
        raise NotImplementedError

class ContextSwitchingTracker(AbstractStateTracker):
    _capture_components = CoverageStateTracker._capture_components | \
        InferenceTracker._capture_components
    _capture_paths = CoverageStateTracker._capture_paths + \
        InferenceTracker._capture_paths

    def __init__(self, *args, **kwargs):
        # properties
        self.mode = InferenceMode.Discovery
        self.cov_tracker = CoverageStateTracker(*args, **kwargs)
        self.inf_tracker = InferenceTracker(*args, **kwargs)
        self.capability_matrix = np.empty((0,0), dtype=object)
        self.nodes_seq = np.empty((0,), dtype=object)
        self.equivalence_map = {}

    async def initialize(self):
        await super().initialize()
        await self.cov_tracker.initialize()
        await self.inf_tracker.initialize()

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return config['tracker'].get('type') == 'inference'

    def __getattribute__(self, name):
        if name in ('update_state', 'update_transition', 'entry_state',
                'current_state', 'state_graph', 'peek', 'reset_state'):
            return getattr(self.current_tracker, name)
        return super().__getattribute__(name)

    @property
    def current_tracker(self):
        match self.mode:
            case InferenceMode.Discovery:
                return self.cov_tracker
            case _:
                return self.inf_tracker

    @property
    def unmapped_states(self):
        G = self.cov_tracker.state_graph
        nodes = G.nodes
        return nodes - self.equivalence_map.keys()

    ## Empty methods that will never be accessed

    def update_state(self, state: AbstractState, *,
            input: AbstractInput, exc: Exception=None, **kwargs) -> Any:
        raise NotImplementedError
    def update_transition(self, source: AbstractState,
            destination: AbstractState, input: AbstractInput, *,
            state_changed: bool, exc: Exception=None, **kwargs) -> Any:
        raise NotImplementedError

    @property
    def entry_state(self) -> AbstractState:
        raise NotImplementedError

    @property
    def current_state(self) -> AbstractState:
        raise NotImplementedError

    @property
    def state_graph(self) -> AbstractStateGraph:
        raise NotImplementedError

    def peek(self, default_source: AbstractState, expected_destination: AbstractState) -> AbstractState:
        raise NotImplementedError

    def reset_state(self, state: AbstractState):
        raise NotImplementedError

class StateInferenceStrategy(UniformStrategy,
        capture_components={'tracker'},
        capture_paths=['strategy.inference_threshold']):
    def __init__(self, *, tracker: ContextSwitchingTracker,
            inference_threshold: Optional[str]=None, **kwargs):
        super().__init__(**kwargs)
        self._tracker = tracker
        self._discovery_threshold = int(inference_threshold or '50')
        TimeElapsedProfiler('time_crosstest').toggle()

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return config['strategy'].get('type') == 'inference'

    async def step(self, input: Optional[AbstractInput]=None):
        match self._tracker.mode:
            case InferenceMode.Discovery:
                if len(self._tracker.unmapped_states) > self._discovery_threshold:
                    self._tracker.mode = InferenceMode.CrossPollination
                else:
                    await super().step(input)
            case InferenceMode.CrossPollination:
                TimeElapsedProfiler('time_crosstest').toggle()
                cap, eqv_map, mask, nodes = await self.perform_cross_pollination()
                self._tracker.capability_matrix = cap[mask,:][:,mask]
                self._tracker.equivalence_map = eqv_map
                self._tracker.nodes_seq = nodes

                collapsed = self._collapse_graph(cap[~mask,:][:,~mask])
                self._tracker.inf_tracker.reconstruct_graph(collapsed)
                self._tracker.mode = InferenceMode.Discovery
                TimeElapsedProfiler('time_crosstest').toggle()

    @classmethod
    def _collapse_graph(cls, adj):
        last_adj = adj
        while True:
            # construct equivalence sets
            stilde = cls._construct_equivalence_sets(adj, dual_axis=True)
            # remove strictly subsumed nodes
            adj, stilde, node_mask, sub_map = cls._eliminate_subsumed_nodes(adj,
                stilde, dual_axis=True)
            # collapse capability matrix where equivalence states exist
            adj, eqv_map, node_mask = cls._collapse_adj_matrix(adj, stilde,
                node_mask, sub_map)
            # mask out collapsed nodes
            adj = adj[~node_mask,:][:,~node_mask]
            if adj.shape == last_adj.shape:
                break
            debug(f"Reduced from {last_adj.shape} to {adj.shape}")
            last_adj = adj
        return last_adj

    @staticmethod
    def intersect1d_nosort(a, b, /):
        idx = np.indices((*a.shape, *b.shape))
        equals = lambda i, j: a[i] == b[j]
        equals_ufn = np.frompyfunc(equals, 2, 1)
        match = equals_ufn(*idx)
        a_idx, b_idx = np.where(match)
        return a_idx, b_idx

    async def perform_cross_pollination(self):
        G = self._tracker.cov_tracker.state_graph
        nodes = np.array(G.nodes)

        to_idx, from_idx = self.intersect1d_nosort(nodes, self._tracker.nodes_seq)

        # get current adjacency matrix
        adj = G.adjacency_matrix
        # and current capability matrix
        cap = self._tracker.capability_matrix
        try:
            assert cap.shape == (len(from_idx), len(from_idx))
        except AssertionError:
            import ipdb; ipdb.set_trace()

        # mask out edges which have already been cross-tested
        edge_mask = adj != None
        mask_irow, mask_icol = np.meshgrid(to_idx, to_idx, indexing='ij')
        edge_mask[mask_irow, mask_icol] = True

        # get a new capability matrix, overlayed with new adjacencies
        cap = self._overlay_capabilities(cap, from_idx, adj, to_idx)

        # get a capability matrix extended with cross-pollination
        await self._extend_cap_matrix(cap, nodes, edge_mask)

        # construct equivalence sets
        stilde = self._construct_equivalence_sets(cap)

        # remove strictly subsumed nodes
        cap, stilde, node_mask, sub_map = self._eliminate_subsumed_nodes(cap, stilde)

        # collapse capability matrix where equivalence states exist
        cap, eqv_map, node_mask = self._collapse_adj_matrix(cap, stilde, node_mask, sub_map)

        assert len(eqv_map) == len(nodes)

        # translate indices back to nodes
        for i in np.where(node_mask)[0]:
            eqv_map.setdefault(i, -1)
        eqv_map = {nodes[l]: s_idx for l, s_idx in eqv_map.items()}

        return cap, eqv_map, node_mask, nodes

    @staticmethod
    def _overlay_capabilities(cap, from_idx, adj, to_idx):
        from_irow, from_icol = np.meshgrid(from_idx, from_idx, indexing='ij')
        to_irow, to_icol = np.meshgrid(to_idx, to_idx, indexing='ij')
        adj[to_irow, to_icol] = cap[from_irow, from_icol]
        return adj

    async def _extend_cap_matrix(self, cap, nodes, edge_mask):
        init_done = np.count_nonzero(edge_mask)
        init_pending = edge_mask.size - init_done
        for src_idx, src in enumerate(nodes):
            for dst_idx, inputs in filter(lambda t: t[1],
                    enumerate(cap[src_idx,:])):
                dst = nodes[dst_idx]
                for eqv_idx, eqv_node in filter(lambda t: t[1] != src,
                        enumerate(nodes)):
                    if edge_mask[eqv_idx, dst_idx]:
                        # this edge has already been tested
                        continue
                    for inp in inputs:
                        try:
                            if not await self._perform_one_cross_pollination(
                                    eqv_node, dst, inp):
                                break
                        except Exception:
                            break
                    else:
                        # eqv_node matched all the responses of src to reach dst
                        self._update_cap_matrix(cap, eqv_idx, dst_idx, inputs)
                    edge_mask[eqv_idx, dst_idx] = True

                    current_done = np.count_nonzero(edge_mask) - init_done
                    percent = f'{100*current_done/init_pending:.1f}%'
                    ValueProfiler('status')(f'cross_test ({percent})')

    async def _perform_one_cross_pollination(self, eqv_src: AbstractState,
            eqv_dst: AbstractState, input: AbstractInput):
        # hacky for now; switch back to coverage states temporarily
        restore_mode = self._tracker.mode
        self._tracker.mode = InferenceMode.Discovery

        try:
            assert eqv_src == await self._explorer.reload_state(eqv_src,
                dryrun=True)
            await self._explorer.loader.execute_input(input)
            # TODO add new states to graph and match against them too?
            current_state = self._tracker.cov_tracker.peek(eqv_src, eqv_dst,
                do_not_cache=True)
            return current_state == eqv_dst
        finally:
            self._tracker.mode = restore_mode

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
        for subv in svee:
            for u in subv:
                for eqv in stilde:
                    if u in eqv:
                        # discard u from all eqv sets in S~
                        eqv.discard(u)
                        assert mask[u]

        # discard the empty set if it exists
        stilde = {frozenset(x) for x in stilde}
        stilde.discard(set())
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
            try:
                adj = np.vstack((adj, out_edges))
                adj = np.hstack((adj, np.vstack((in_edges, self_edges))))
            except Exception:
                import ipdb; ipdb.set_trace()

            for l in idx:
                eqv_map[l] = s_idx
                eqv_mask[l] = True
            s_idx += 1

        return adj, eqv_map, eqv_mask

class InferenceWebRenderer(WebRenderer):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return config['strategy'].get('type') == 'inference'

    async def _handle_websocket(self, request):
        async def _handler():
            ws = web.WebSocketResponse(compress=False)
            await ws.prepare(request)

            data_loader = InferenceWebDataLoader(ws, self._session, self._hitcounter)
            gather_tasks = asyncio.gather(*data_loader.tasks)
            try:
                await gather_tasks
            except (RuntimeError, ConnectionResetError) as ex:
                debug(f'Websocket handler terminated ({ex=})')
            finally:
                gather_tasks.cancel()
        await get_session_task_group().create_task(_handler())

class InferenceWebDataLoader(WebDataLoader):
    @property
    def fresh_graph(self):
        H = self._session._explorer.tracker.inf_tracker.state_graph
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