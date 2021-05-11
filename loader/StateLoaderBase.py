from abc          import ABC,abstractmethod
from common       import LoadedException
from fuzzer       import Environment,
                         ChannelFactoryBase,
                         ChannelBase
from input        import InputBase,
                         PreparedInput
from statemanager import StateBase,
                         StateManager

class StateLoaderBase(ABC):
    """
    The state loader maintains a running process and provides the capability of
    switching program states.
    """
    def __init__(self, exec_env: Environment, ch_env: ChannelFactoryBase):
        self._exec_env = exec_env
        self._ch_env = ch_env

    @abstractmethod
    def load_state(self, state: StateBase):
        pass

    def execute_input(self, input: InputBase, channel: ChannelBase)
        -> PreparedInput:
        """
        Executes the sequence of interactions specified by the input.

        :param      input:  An object derived from the InputBase abstract class.
        :type       input:  InputBase

        :param      channel: The communication channel over which the interaction
                    is performed. This is usually provided by the loader.
        :type       channel: ChannelBase

        :returns:   The sequence of interactions actually performed
        :rtype:     PreparedInput

        :raises:    LoadedException: Forwards the exception that was raised
                    during execution, along with the input that caused it.
        """
        prep_input = PreparedInput()
        for interaction in input:
            try:
                # TODO figure out what other parameters this needs
                performed = interaction.perform(channel)
                prep_input.extend(performed)
            except Exception as ex:
                prep_input.append(interaction)
                raise LoadedException(ex, prep_input)

            try:
                # TODO collect state info
                # TODO perform fault detection
                # TODO update state machine
                # TODO poll channel for incoming data? or do it async?
                pass
            except Exception as ex:
                raise LoadedException(ex, prep_input)

        return prep_input