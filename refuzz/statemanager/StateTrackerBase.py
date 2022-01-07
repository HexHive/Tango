from abc          import ABC, abstractmethod
from typing       import Callable
from statemanager import StateBase
from input        import InputBase
from generator    import InputGeneratorBase

class StateTrackerBase(ABC):
    def __init__(self, generator: InputGeneratorBase):
        # a state tracker might need the generator for a training phase
        # FIXME maybe make this specific to the state tracker that needs it?
        self._generator = generator

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
    def update(self, prev: StateBase, new: StateBase,
            input_gen: Callable[..., InputBase]):
        pass