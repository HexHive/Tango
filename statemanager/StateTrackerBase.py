from abc          import ABC,      abstractmethod
from statemanager import StateBase

class StateTrackerBase(ABC):
    @property
    @abstractmethod
    def initial_state(self) -> StateBase:
        pass

    @property
    @abstractmethod
    def current_state(self) -> StateBase:
        pass
