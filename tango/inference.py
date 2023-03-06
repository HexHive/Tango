from __future__ import annotations

from . import debug, info, warning

from tango.core import (UniformStrategy, AbstractState, AbstractInput,
    BaseStateGraph, AbstractStateTracker)
from tango.cov import CoverageStateTracker

from typing import Optional
from enum import Enum, auto
from nptyping import NDArray, Shape
import numpy as np
import networkx as nx

__all__ = ['StateInferenceStrategy', 'StateInferenceTracker']

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
        self._cov_tracker = CoverageStateTracker(*args, **kwargs)
        self._inf_tracker = InferenceTracker(*args, **kwargs)
        self._mode = InferenceMode.Discovery
        self._eqv_map = {}

    async def initialize(self):
        await super().initialize()
        await self._cov_tracker.initialize()
        await self._inf_tracker.initialize()

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return config['tracker'].get('type') == 'inference'

    def __getattribute__(self, name):
        if name in ('update_state', 'update_transition', 'entry_state',
                'current_state', 'state_graph', 'peek', 'reset_state'):
            return getattr(self.current_tracker, name)
        return super().__getattribute__(name)

    @property
    def mode(self) -> InferenceMode:
        return self._mode

    @mode.setter
    def mode(self, value: InferenceMode):
        self._mode = value

    @property
    def current_tracker(self):
        match self._mode:
            case InferenceMode.Discovery:
                return self._cov_tracker
            case _:
                return self._inf_tracker

    @property
    def equivalence_map(self):
        return self._eqv_map

    @equivalence_map.setter
    def equivalence_map(self, eqv_map):
        self._eqv_map = eqv_map

    @property
    def unmapped_states(self):
        G = self._cov_tracker.state_graph
        nodes = G.nodes
        return nodes - self._eqv_map.keys()

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
        capture_components={'tracker'}):
    def __init__(self, *, tracker: ContextSwitchingTracker, **kwargs):
        super().__init__(**kwargs)
        self._tracker = tracker
        self._discovery_threshold = 50

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return config['strategy'].get('type') == 'inference'

    async def step(self, input: Optional[AbstractInput]=None):
        match self._tracker.mode:
            case InferenceMode.Discovery:
                if len(self._tracker.unmapped_states) > self._discovery_threshold:
                    warning(f'Unmapped: {self._tracker.unmapped_states}')
                    self._tracker.mode = InferenceMode.CrossPollination
                else:
                    await super().step(input)
            case InferenceMode.CrossPollination:
                cap, eqv_map = await self.perform_cross_pollination()
                self._tracker.equivalence_map = eqv_map
                self._tracker.current_tracker.reconstruct_graph(cap)
                self._tracker._mode = InferenceMode.Discovery
                warning(eqv_map)

    async def perform_cross_pollination(self):
        G = self._tracker._cov_tracker.state_graph
        nodes = list(G.nodes)

        # get a capability matrix extended with cross-pollination
        cap = await self._extend_adj_matrix(G)

        # construct equivalence sets
        stilde = self._construct_equivalence_sets(cap)

        # remove strictly subsumed nodes
        cap, stilde, mask = self._eliminate_subsumed_nodes(cap, stilde)

        # collapse capability matrix where equivalence states exist
        cap, eqv_map, mask = self._collapse_cap_matrix(cap, stilde, mask)

        # translate indices back to nodes
        for i in np.where(mask)[0]:
            eqv_map.setdefault(i, -1)
        eqv_map = {nodes[l]: s_idx for l, s_idx in eqv_map.items()}

        return cap, eqv_map

    async def _extend_adj_matrix(self, G):
        nodes = list(G.nodes)
        adj, cap = G.adjacency_matrix, G.adjacency_matrix

        for src_idx, src in enumerate(nodes):
            for dst_idx, inputs in filter(lambda t: t[1],
                    enumerate(adj[src_idx,:])):
                dst = nodes[dst_idx]
                for eqv_idx, eqv_node in filter(lambda t: t[1] != src,
                        enumerate(nodes)):
                    for inp in inputs:
                        try:
                            if not await self._perform_one_cross_pollination(
                                    eqv_node, dst, inp):
                                break
                        except Exception:
                            break
                    else:
                        # eqv_node matched all the responses of src to reach dst
                        self._update_adj_matrix(cap, eqv_idx, dst_idx, inputs)
        return cap

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
            current_state = self._tracker.current_tracker.peek(eqv_src, eqv_dst,
                do_not_cache=True)
            return current_state == eqv_dst
        finally:
            self._tracker.mode = restore_mode

    @staticmethod
    def _update_adj_matrix(adj, src_idx, dst_idx, inputs):
        row = adj[src_idx,:]
        if not (t := row[dst_idx]):
            row[dst_idx] = list(inputs)
        else:
            warning("Appending to existing transition; this should not happen"
                " if the original graph is a tree.")
            row[dst_idx] = list(t) + list(inputs)

    @staticmethod
    def _construct_equivalence_sets(cap, dual_axis=False):
        if dual_axis:
            raise NotImplementedError
        axis = 0

        stilde = set()
        is_nbr = cap != None

        idx = np.array(range(is_nbr.shape[axis]))
        _, membership = np.unique(is_nbr, axis=axis, return_inverse=True)
        order = np.argsort(membership)
        _, split = np.unique(membership[order], return_index=True)
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
    def _eliminate_subsumed_nodes(cls, cap, stilde, dual_axis=False):
        if dual_axis:
            raise NotImplementedError

        svee = set()
        is_nbr = cap != None
        nbrs_fn = lambda x: frozenset(np.where(x)[0])
        nbrs = np.apply_along_axis(nbrs_fn, 1, is_nbr)

        for v, nbv in enumerate(nbrs):
            subv = set()
            for u, nbu in enumerate(nbrs):
                if nbv > nbu:
                    subv.add(u)
            if subv:
                svee.add(frozenset(subv))

        mask = np.zeros(len(cap), dtype=bool)
        stilde = [set(x) for x in stilde]
        for subv in svee:
            for u in subv:
                for eqv in stilde:
                    if u in eqv:
                        # discard u from all eqv sets in S~
                        eqv.discard(u)
                        # mark for deletion from cap matrix
                        mask[u] = True

                        # FIXME transitions of subsumed nodes need to be
                        # substitued in subsumers, especially when subsumption
                        # not dual_axis, because subsumers are only counted
                        # along the outward axis...

        # discard the empty set if it exists
        stilde = {frozenset(x) for x in stilde}
        stilde.discard(set())
        # cls._adjust_stilde(stilde, mask)
        # cap = cap[~mask,:][:,~mask]
        cap[mask,:] = None
        cap[:,mask] = None
        return cap, stilde, mask

    @classmethod
    def _collapse_cap_matrix(cls, cap, stilde, mask):
        def equal_unordered(s, t):
            t = list(t)   # make a mutable copy
            try:
                for elem in s:
                    t.remove(elem)
            except ValueError:
                return False
            return not t

        def combine_transitions(r, t):
            if not r:
                return
            elif not t:
                return r
            combined = list(r)
            for elem in t:
                if not elem in combined:
                    combined.append(elem)
            return combined
        combine_transitions = np.frompyfunc(combine_transitions, 2, 1,
            identity=[]).reduce

        eqv_mask = np.concatenate((mask, np.zeros(len(stilde), dtype=bool)))
        eqv_map = {}
        s_idx = 0
        for eqv in stilde:
            idx = np.array(list(eqv))
            out_edges = combine_transitions(cap[idx,:], axis=0)
            in_edges = combine_transitions(cap[:,idx], axis=1).reshape(cap.shape[0], 1)
            self_edges = combine_transitions(cap[idx,idx], axis=0, keepdims=True)
            self_edges = combine_transitions(self_edges, keepdims=True)
            # s_idx = cap.shape[0]
            try:
                cap = np.vstack((cap, out_edges))
                cap = np.hstack((cap, np.vstack((in_edges, self_edges))))
            except Exception:
                import ipdb; ipdb.set_trace()

            for l in idx:
                eqv_map[l] = s_idx
                eqv_mask[l] = True
            s_idx += 1

        cap = cap[~eqv_mask,:][:,~eqv_mask]
        assert cap.shape == (len(stilde), len(stilde))
        return cap, eqv_map, mask
