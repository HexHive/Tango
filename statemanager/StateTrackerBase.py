from abc          import ABC, abstractmethod
from statemanager import StateBase, TransitionBase
from input        import InputBase

class StateTrackerBase(ABC):
    @property
    @abstractmethod
    def initial_state(self) -> StateBase:
        pass

    @property
    @abstractmethod
    def initial_transition(self) -> TransitionBase:
        pass

    @property
    @abstractmethod
    def current_state(self) -> StateBase:
        pass

    @abstractmethod
    def update_state(self, state: StateBase, input: InputBase):
        pass