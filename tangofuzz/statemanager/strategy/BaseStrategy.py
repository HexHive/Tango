from statemanager.strategy import AbstractStrategy
from input import AbstractInput
from statemanager import StateMachine
from tracker import AbstractState

class BaseStrategy(AbstractStrategy):
    def __init__(self, *, sm: StateMachine, entry_state: AbstractState):
        self._sm = sm
        self._entry = entry_state

    def update_state(self, state: AbstractState, *, input: AbstractInput, exc: Exception=None, **kwargs):
        pass

    def update_transition(self, source: AbstractState, destination: AbstractState, input: AbstractInput, *, state_changed: bool, exc: Exception=None, **kwargs):
        pass