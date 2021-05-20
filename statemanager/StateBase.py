from abc          import ABC, abstractmethod
from statemanager import StateManager, TransitionBase
from input        import InputBase

class StateBase(ABC):
    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def get_escaper(self) -> InputBase:
        pass

    @abstractmethod
    def update(self, sman: StateManager, transition: TransitionBase):
        pass