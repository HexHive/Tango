from statemanager import StateTrackerBase
from statemanager import GrammarState, GrammarTransition
from loader import StateLoaderBase
from input import GrammarInput


class GrammarStateTracker(StateTrackerBase):
    """
    State Tracker class for Grammar State Machine.
    Might not be used too much but for coherence right now.
    """

    def __init__(self, loader: StateLoader):
        pass

    @property
    def inital_state(self) -> GrammarState:
        pass
    
    @property
    def initial_transition(self) -> GrammarTransition:
        pass

    @property
    def current_state(self) -> GrammarState:
        pass

    @property
    def update_state(self, state: GrammarState, input: GrammarInput):
        pass      
