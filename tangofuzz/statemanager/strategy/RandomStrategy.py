from statemanager import StateBase, StrategyBase, StateMachine
from random import Random
from profiler import ProfileValue

class RandomStrategy(StrategyBase):
    def __init__(self, entropy: Random, limit: int = 100, **kwargs):
        super().__init__(**kwargs)
        self._entropy = entropy
        self._counter = 0
        self._limit = limit
        self._invalid_states = set()
        self._target_state = self._entry

    def _recalculate_target(self):
        self._counter = 0
        ProfileValue('strat_counter')(self._counter)
        filtered = [x for x in self._sm._graph.nodes if x not in self._invalid_states]
        self._target_state = self._entropy.choice(filtered)

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

    def update_transition(self, source: StateBase, destination: StateBase, invalidate: bool = False):
        pass