from abc          import ABC,abstractmethod
from common       import LoadedException
from loader       import Environment
from networkio    import (ChannelFactoryBase,
                         ChannelBase)
from input        import (InputBase,
                         PreparedInput)
from statemanager import (StateBase,
                         StateManager)
from interaction  import ReceiveInteraction

from profiler import ProfileFrequency, ProfileValueMean, ProfileEvent, ProfileCount

class StateLoaderBase(ABC):
    """
    The state loader maintains a running process and provides the capability of
    switching program states.
    """
    def __init__(self, exec_env: Environment, ch_env: ChannelFactoryBase):
        self._exec_env = exec_env
        self._ch_env = ch_env

    @abstractmethod
    def load_state(self, state: StateBase, sman: StateManager):
        pass

    @property
    @abstractmethod
    def channel(self):
        pass

    @ProfileEvent("execute_input")
    @ProfileFrequency("execs")
    def execute_input(self, input: InputBase, sman: StateManager):
        """
        Executes the sequence of interactions specified by the input.

        :param      input:  An object derived from the InputBase abstract class.
        :type       input:  InputBase

        :raises:    LoadedException: Forwards the exception that was raised
                    during execution, along with the input that caused it.
        """
        with sman.get_context(input) as ctx:
            try:
                idx = -1
                for idx, interaction in enumerate(ctx):
                    # FIXME figure out what other parameters this needs
                    interaction.perform(self.channel)
                    # TODO perform fault detection
                else:
                    ProfileValueMean("input_len", samples=100)(idx + 1)
                    ProfileCount("interactions")(idx + 1)
            except Exception as ex:
                raise LoadedException(ex, ctx.input_gen())

        # poll channel for incoming data
        # FIXME what to do with this data? if input did not request it, it was
        #   probably useless
        data = self.channel.receive()