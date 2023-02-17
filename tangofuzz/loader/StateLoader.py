from abc          import ABC, abstractmethod
from typing       import Union
from input        import AbstractInput
# from statemanager import StateManager # FIXME there seems to be a cyclic dep
from tracker import AbstractState
from profiler import FrequencyProfiler, EventProfiler

class AbstractStateLoader(ABC):
    """
    The state loader maintains a running target and provides the capability of
    switching program states.
    """
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        fn = EventProfiler('execute_input')(
            FrequencyProfiler('execs', period=1)(cls.execute_input))
        setattr(cls, 'execute_input', fn)

    @abstractmethod
    async def load_state(self, state_or_path: Union[AbstractState, list], sman) -> AbstractState:
        pass

    @abstractmethod
    async def execute_input(self, input: AbstractInput):
        pass

    @property
    @abstractmethod
    def channel(self):
        pass