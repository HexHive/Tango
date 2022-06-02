from statemanager import StateBase, StrategyBase, StateMachine
from random import Random
from profiler import ProfileValue
from input import InputBase, ZoomInput
from interaction import ReachInteraction
import networkx as nx

class ZoomStrategy(StrategyBase):
    def __init__(self, entropy: Random, limit: int=100, **kwargs):
        super().__init__(**kwargs)
        self._entropy = entropy
        self._counter = 0
        self._limit = limit
        self._invalid_states = set()
        self._target_state = self._entry
        self._cycles = 0

    @staticmethod
    def _calculate_edge_weight(src, dst, data):
        return ReachInteraction.l2_distance(
            (src._struct.x, src._struct.y),
            (dst._struct.x, dst._struct.y)
        )

    def _recalculate_target(self):
        if all(x._cycle == self._cycles for x in self._sm._graph.nodes):
            self._cycles += 1
        self._counter = 0
        ProfileValue('strat_counter')(self._counter)
        filtered = [x for x in self._sm._graph.nodes if x not in self._invalid_states]
        distances = nx.shortest_path_length(self._sm._graph, self._entry, \
                                        weight=self._calculate_edge_weight, \
                                        method='bellman-ford')
        self._target_state = next(x for x in sorted(
            distances.keys(),
            key=lambda x: distances[x],
            reverse=True
        ) if x._cycle < self._cycles)
        self._target_state._cycle = self._cycles

    def step(self) -> bool:
        should_reset = False
        if self._counter == 0:
            should_reset = True
            self._recalculate_target()
        else:
            ProfileValue('strat_counter')(self._counter)
        self._counter += 1
        self._counter %= self._limit
        return should_reset

    @property
    def target_state(self) -> StateBase:
        return self._target_state

    @property
    def target(self) -> StateBase:
        return self._target_state

    def update_state(self, state: StateBase, invalidate: bool = False, is_new: bool = False):
        if invalidate:
            self._invalid_states.add(state)
            if self._target_state == state:
                self._recalculate_target()
        else:
            self._invalid_states.discard(state)
            # any state we reach in the current cycle is counted as visited
            state._cycle = self._cycles

    def update_transition(self, source: StateBase, destination: StateBase, input: InputBase, invalidate: bool = False):
        src_x, src_y, src_z = map(lambda c: getattr(source._struct, c), ('x', 'y', 'z'))
        dst_x, dst_y, dst_z = map(lambda c: getattr(destination._struct, c), ('x', 'y', 'z'))
        if dst_z > src_z or abs(src_z - dst_z) <= 16:
            # FIXME this interaction is not used, since it is replaced by the loader
            inp = ZoomInput([ReachInteraction(destination, (src_x, src_y, src_z))])
            self._sm.update_transition(destination, source, inp)