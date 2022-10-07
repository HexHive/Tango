from abc          import ABC, abstractmethod
from typing       import Callable
from tracker import StateBase
from input        import InputBase
from loader       import StateLoaderBase

class StateTrackerBase(ABC):
    @abstractmethod
    async def create(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def entry_state(self) -> StateBase:
        """
        The state of the target when it is first launched (with no inputs sent)

        :returns:   The state object describing the entry state.
        :rtype:     StateBase
        """
        pass

    @property
    @abstractmethod
    def current_state(self) -> StateBase:
        pass

    @abstractmethod
    def update(self, source: StateBase, destination: StateBase, input_gen: Callable[..., InputBase], dryrun: bool=False) -> StateBase:
        pass

    def reset_state(self, state: StateBase):
        """
        Informs the state tracker that the loader has reset the target into a
        state.
        """
        pass

    @property
    def state_manager(self):
        return self._sman

    @state_manager.setter
    def state_manager(self, sman):
        self._sman = sman