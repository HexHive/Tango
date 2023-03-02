from __future__ import annotations

from . import debug, info, warning

from tango.core import UniformStrategy, AbstractState, AbstractInput
from tango.cov import LoaderDependentTracker, CoverageStateTracker

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

class StateInferenceTracker(LoaderDependentTracker,
        capture_paths=['tracker.native_lib']):
    def __init__(self, *, native_lib: str=None, **kwargs):
        super().__init__(**kwargs)
        self._cov_tracker = CoverageStateTracker(native_lib=native_lib,
            loader=self._loader)
        self._mode = InferenceMode.Discovery

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return config['tracker'].get('type') == 'inference'

    async def initialize(self):
        await self._cov_tracker.initialize()
        await super().initialize()

    @property
    def state_graph(self):
        match self._mode:
            case InferenceMode.Discovery:
                return self._cov_tracker.state_graph
            case _:
                pass

    @property
    def entry_state(self) -> AbstractState:
        match self._mode:
            case InferenceMode.Discovery:
                return self._cov_tracker.entry_state
            case _:
                pass

    @property
    def current_state(self) -> AbstractState:
        match self._mode:
            case InferenceMode.Discovery:
                return self._cov_tracker.current_state
            case _:
                pass

    def peek(self, default_source: AbstractState, expected_destination: AbstractState) -> AbstractState:
        match self._mode:
            case InferenceMode.Discovery:
                return self._cov_tracker.peek(default_source, expected_destination)
            case _:
                pass

    def reset_state(self, state: AbstractState):
        match self._mode:
            case InferenceMode.Discovery:
                return self._cov_tracker.reset_state(state)
            case _:
                pass

    def update_state(self, source: AbstractState, /, *, input: AbstractInput,
            exc: Exception=None, peek_result: Optional[AbstractState]=None) \
            -> Optional[AbstractState]:
        match self._mode:
            case InferenceMode.Discovery:
                return self._cov_tracker.update_state(source, input=input,
                    exc=exc, peek_result=peek_result)
            case _:
                pass

    def update_transition(self, source: AbstractState,
            destination: AbstractState, input: AbstractInput, *,
            state_changed: bool, exc: Exception=None):
        match self._mode:
            case InferenceMode.Discovery:
                return self._cov_tracker.update_transition(source, destination,
                    input, state_changed=state_changed, exc=exc)
            case _:
                pass

    @property
    def mode(self) -> InferenceMode:
        return self._mode

    @mode.setter
    def mode(self, value: InferenceMode):
        self._mode = value

class StateInferenceStrategy(UniformStrategy,
        capture_components={'tracker'}):
    def __init__(self, *, tracker: StateInferenceTracker, **kwargs):
        super().__init__(**kwargs)
        self._tracker = tracker
        self._discovery_threshold = 10

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return config['strategy'].get('type') == 'inference'

    async def step(self, input: Optional[AbstractInput]=None):
        match self._tracker.mode:
            case InferenceMode.Discovery:
                if len(self._tracker.state_graph) > self._discovery_threshold:
                    self._tracker.mode = InferenceMode.CrossPollination
                else:
                    await super().step(input)
            case InferenceMode.CrossPollination:
                await self.perform_cross_pollination()

    async def perform_cross_pollination(self):
        G = self._tracker.state_graph

        # get an adj matrix extended with cross-pollination
        adj = self._extend_adj_matrix(G)

        # collapse adjacency matrix where super-equivalence states exist
        pass

    def _extend_adj_matrix(self, G):
        nodes = G.nodes
        adj, eqv_adj = G.adjacency_matrix, G.adjacency_matrix

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
                        self._update_adj_matrix(eqv_adj, eqv_idx, dst_idx, inputs)

        # TODO do we need to do multiple passes until matrix settles?
        return eqv_adj



    async def _perform_one_cross_pollination(self, eqv_src: AbstractState,
            eqv_dst: AbstractState, input: AbstractInput):
        # hacky for now; switch back to coverage states temporarily
        restore_mode = self._tracker.mode
        self._tracker.mode = InferenceMode.Discovery

        try:
            assert eqv_src == await self._explorer.reload_state(eqv_src,
                dryrun=True)
            await self._explorer.loader.execute_input(input)
            current_state = self._tracker.peek(eqv_src, eqv_dst)
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
    def _collapse_adj_matrix(G, adj):
        def equal_nohash(s, t):
            t = list(t)   # make a mutable copy
            try:
                for elem in s:
                    t.remove(elem)
            except ValueError:
                return False
            return not t

        nodes = G.nodes
        groupings = {i: {i} for i in range(len(nodes))}
        updated = False
        while True:
            for src_idx, src in enumerate(nodes):
                eqv_set = set(groupings.keys())
                excl_set = set()
                for dst_idx, inputs in filter(lambda t: t[1],
                        enumerate(adj[src_idx,:])):
                    # t[0] != src_idx and
                    in_set = set(map(lambda t: t[0],
                        filter(lambda t: t[1] and \
                            equal_nohash(t[1], adj[src_idx, dst_idx]),
                            enumerate(adj[:,dst_idx]))
                        ))
                    eqv_set &= in_set
                    excl_set |= in_set
                assert src_idx in eqv_set and src_idx in excl_set
                if eqv_set == excl_set:
                    for idx in eqv_set:
                        groupings[idx] |= eqv_set
                        updated = True
            if not updated:
                break

class EquivalenceAdjacencyMatrix()