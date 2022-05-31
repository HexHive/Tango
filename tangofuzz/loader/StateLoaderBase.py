from . import critical, debug
from abc          import ABC, abstractmethod
from typing       import Union
from common       import LoadedException, ChannelBrokenException, async_enumerate
from loader       import Environment
from networkio    import (ChannelFactoryBase,
                         ChannelBase)
from input        import (InputBase,
                         PreparedInput)
from statemanager import (StateBase,
                         StateManager)
from interaction  import ReceiveInteraction
from profiler import ProfileFrequency, ProfileValueMean, ProfileEvent, ProfileCount
import ctypes

from ptrace import PtraceError

class StateLoaderBase(ABC):
    """
    The state loader maintains a running process and provides the capability of
    switching program states.
    """
    def __init__(self, exec_env: Environment, ch_env: ChannelFactoryBase,
            no_aslr: bool):
        self._exec_env = exec_env
        self._ch_env = ch_env

        if no_aslr:
            ADDR_NO_RANDOMIZE = 0x0040000
            personality = ctypes.CDLL(None).personality
            personality.restype = ctypes.c_int
            personality.argtypes = [ctypes.c_ulong]
            personality(ADDR_NO_RANDOMIZE)

    @abstractmethod
    async def load_state(self, state_or_path: Union[StateBase, list], sman: StateManager, update: bool):
        pass

    @property
    @abstractmethod
    def channel(self):
        pass

    @ProfileEvent("execute_input")
    @ProfileFrequency("execs")
    async def execute_input(self, input: InputBase, sman: StateManager, update: bool = True):
        """
        Executes the sequence of interactions specified by the input.

        :param      input:  An object derived from the InputBase abstract class.
        :type       input:  InputBase

        :raises:    LoadedException: Forwards the exception that was raised
                    during execution, along with the input that caused it.
        """
        if update:
            with sman.get_context(input) as ctx:
                try:
                    idx = -1
                    async for idx, interaction in async_enumerate(ctx):
                        # FIXME figure out what other parameters this needs
                        debug(interaction)
                        await interaction.perform(self.channel)
                        # TODO perform fault detection
                    else:
                        ProfileValueMean("input_len", samples=100)(idx + 1)
                        ProfileCount("total_interactions")(idx + 1)
                except Exception as ex:
                    raise LoadedException(ex, ctx.input_gen())
        else:
            idx = -1
            for idx, interaction in enumerate(input):
                try:
                    debug(interaction)
                    await interaction.perform(self.channel)
                except Exception as ex:
                    raise LoadedException(ex, input[:idx + 1])
            else:
                ProfileCount("total_interactions")(idx + 1)

        # poll channel for incoming data
        # FIXME what to do with this data? if input did not request it, it was
        #   probably useless
        try:
            # server may have killed the connection at this point and we need to
            # report it
            # FIXME this should now be addressed through asyncio task cancelling
            data = await self.channel.receive()
        except ChannelBrokenException as ex:
            raise LoadedException(ex, input)
