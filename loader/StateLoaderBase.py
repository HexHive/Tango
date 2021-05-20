from abc          import ABC,abstractmethod
from common       import LoadedException
from loader       import Environment
from networkio    import (ChannelFactoryBase,
                         ChannelBase)
from input        import (InputBase,
                         PreparedInput)
from statemanager import (StateBase,
                         StateManager,
                         PreparedTransition)
from interaction  import ReceiveInteraction

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

    def execute_input(self, input: InputBase, channel: ChannelBase,
            sman: StateManager):
        """
        Executes the sequence of interactions specified by the input.

        :param      input:  An object derived from the InputBase abstract class.
        :type       input:  InputBase

        :param      channel: The communication channel over which the interaction
                    is performed. This is usually provided by the loader.
        :type       channel: ChannelBase

        :raises:    LoadedException: Forwards the exception that was raised
                    during execution, along with the input that caused it.
        """
        prep_input = PreparedInput()
        for interaction in input:
            try:
                # FIXME figure out what other parameters this needs
                performed = interaction.perform(channel)
                prep_input.extend(performed)
            except Exception as ex:
                prep_input.append(interaction)
                raise LoadedException(ex, prep_input)

            try:
                # poll channel for incoming data
                data = channel.receive()
                if data:
                    prep_input.append(ReceiveInteraction(data=data))

                # collect state info and update state machine
                if sman.update(PreparedTransition(prep_input)):
                    # if a new state is observed, clear the input
                    prep_input = PreparedInput()

                # TODO perform fault detection
            except Exception as ex:
                raise LoadedException(ex, prep_input)