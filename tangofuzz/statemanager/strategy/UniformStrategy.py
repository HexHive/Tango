from statemanager import StateBase, StrategyBase, StateMachine
from random import Random
from profiler import ProfileValue

class UniformStrategy(StrategyBase):
    def __init__(self, entropy: Random, limit: int=100, **kwargs):
        super().__init__(**kwargs)
        self._entropy = entropy
        self._counter = 0
        self._limit = limit
        self._invalid_states = set()
        self._target_state = self._entry

        self._exp_weights = (0.64, 0.23, 0.09, 0.03, 0.01)
        self._calc_weights = lambda n: (
            self._exp_weights[0] + sum(self._exp_weights[n:]),
            *self._exp_weights[1:n],
            *((0.0,) * (n - len(self._exp_weights))))[:n]

    def update_state(self, state: StateBase, invalidate: bool = False, is_new: bool = False):
        if invalidate:
            self._invalid_states.add(state)
            if self._target_state == state:
                self._recalculate_target()
        else:
            self._invalid_states.discard(state)
            if is_new or not hasattr(state, '_energy'):
                state._energy = 1
            else:
                state._energy += 1

    def _recalculate_target(self):
        filtered = [x for x in self._sm._graph.nodes if x not in self._invalid_states]
        if not filtered:
            self._target_state = self._entry
        else:
            filtered.sort(key=lambda x: x._energy)
            self._target_state = self._entropy.choices(filtered, \
                                    weights=self._calc_weights(len(filtered)), \
                                    k=1)[0]
        for node in self._sm._graph.nodes:
            node._energy = 0

    def update_transition(self, source: StateBase, destination: StateBase, invalidate: bool = False):
        pass

    def step(self) -> bool:
        should_reset = False
        if self._counter == 0:
            old_target = self._target_state
            self._recalculate_target()
            should_reset = old_target != self._target_state
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