from statemanager import StateMachine
from statemanager.strategy import BaseStrategy
from tracker import AbstractState
from random import Random
from profiler import ValueProfiler
from input import AbstractInput

class RandomStrategy(BaseStrategy):
    def __init__(self, entropy: Random, limit: int=100, **kwargs):
        super().__init__(**kwargs)
        self._entropy = entropy
        self._counter = 0
        self._limit = limit
        self._invalid_states = set()
        self._target_state = self._entry

    def _recalculate_target(self):
        self._counter = 0
        ValueProfiler('strat_counter')(self._counter)
        filtered = [x for x in self._sm._graph.nodes if x not in self._invalid_states]
        self._target_state = self._entropy.choice(filtered)

    def step(self) -> bool:
        should_reset = False
        if self._counter == 0:
            should_reset = True
            self._recalculate_target()
        else:
            ValueProfiler('strat_counter')(self._counter)
        self._counter += 1
        self._counter %= self._limit
        return should_reset

    @property
    def target_state(self) -> AbstractState:
        return self._target_state

    @property
    def target(self) -> AbstractState:
        return self._target_state

    def update_state(self, state: AbstractState, *, input: AbstractInput=None, exc: Exception=None, **kwargs):
        if exc:
            self._invalid_states.add(state)
            if self._target_state == state:
                self._recalculate_target()
        else:
            self._invalid_states.discard(state)
