from . import critical, debug
from abc          import ABC, abstractmethod
from typing       import Union
from common       import LoadedException, ChannelBrokenException
from dataio    import ChannelFactoryBase
from input        import InputBase
from generator    import InputGeneratorBase
from statemanager import StateManager # FIXME there seems to be a cyclic dep
from tracker import StateBase, StateTrackerBase
from profiler import ProfileFrequency, ProfileValueMean, ProfileEvent, ProfileCount

class StateLoaderBase(ABC):
    """
    The state loader maintains a running process and provides the capability of
    switching program states.
    """
    def __init__(self, ch_env: ChannelFactoryBase, input_generator: InputGeneratorBase):
        self._ch_env = ch_env
        self._generator = input_generator
        self._tracker = None

    @property
    def state_tracker(self):
        return self._tracker

    @state_tracker.setter
    def state_tracker(self, value: StateTrackerBase):
        self._tracker = value

    @abstractmethod
    async def load_state(self, state_or_path: Union[StateBase, list], sman: StateManager) -> StateBase:
        pass

    @property
    @abstractmethod
    def channel(self):
        pass

    @ProfileEvent("execute_input")
    @ProfileFrequency("execs")
    async def execute_input(self, input: InputBase):
        """
        Executes the sequence of interactions specified by the input.

        :param      input:  An object derived from the InputBase abstract class.
        :type       input:  InputBase

        :raises:    LoadedException: Forwards the exception that was raised
                    during execution, along with the input that caused it.
        """
        try:
            idx = -1
            async for interaction in input:
                idx += 1
                await interaction.perform(self.channel)
        except Exception as ex:
            raise LoadedException(ex, input[:idx + 1]) from ex
        finally:
            ProfileValueMean("input_len", samples=100)(idx + 1)
            ProfileCount("total_interactions")(idx + 1)
