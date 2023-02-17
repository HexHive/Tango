from abc          import ABC, abstractmethod
from typing       import Callable
from tracker import AbstractState
from input        import AbstractInput

class AbstractStateTracker(ABC):
    @property
    def state_manager(self):
        return self._sman

    @state_manager.setter
    def state_manager(self, sman):
        self._sman = sman

    @abstractmethod
    async def create(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def entry_state(self) -> AbstractState:
        """
        The state of the target when it is first launched (with no inputs sent)

        :returns:   The state object describing the entry state.
        :rtype:     AbstractState
        """
        pass

    @property
    @abstractmethod
    def current_state(self) -> AbstractState:
        pass

    @abstractmethod
    def update(self, source: AbstractState, input: AbstractInput) -> AbstractState:
        pass

    @abstractmethod
    def peek(self, default_source: AbstractState=None, expected_destination: AbstractState=None) -> AbstractState:
        pass

    @abstractmethod
    def reset_state(self, state: AbstractState):
        """
        Informs the state tracker that the loader has reset the target into a
        state.
        """
        pass