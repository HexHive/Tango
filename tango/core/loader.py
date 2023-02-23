from tango.core.input        import AbstractInput
from tango.core.tracker import AbstractState
from tango.core.profiler import (FrequencyProfiler, EventProfiler,
                       ValueMeanProfiler, CountProfiler)
from tango.core.dataio    import AbstractChannelFactory, AbstractChannel
from tango.core.types import Transition
from tango.common import AsyncComponent, ComponentType
from tango.exceptions import LoadedException

from abc          import ABC, abstractmethod
from typing       import Union, AsyncGenerator

__all__ = ['AbstractStateLoader', 'BaseStateLoader']

class AbstractStateLoader(AsyncComponent, ABC,
        component_type=ComponentType.loader):
    """
    The state loader maintains a running target and provides the capability of
    switching program states.
    """
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if (target := cls.__dict__.get('execute_input')):
            wrapped = EventProfiler('execute_input')(
                FrequencyProfiler('execs', period=1)(target))
            setattr(cls, 'execute_input', wrapped)

    @abstractmethod
    async def load_state(self, state_or_path: Union[AbstractState, list]) \
            -> AsyncGenerator[Transition, AbstractState]:
        pass

    @abstractmethod
    async def execute_input(self, input: AbstractInput):
        pass

    @property
    @abstractmethod
    def channel(self) -> AbstractChannel:
        pass

class BaseStateLoader(AbstractStateLoader,
        capture_components={ComponentType.channel_factory}):
    def __init__(self, channel_factory: AbstractChannelFactory):
        self._ch_env = channel_factory

    async def execute_input(self, input: AbstractInput):
        """
        Executes the sequence of interactions specified by the input.

        :param      input:  An object derived from the AbstractInput abstract class.
        :type       input:  AbstractInput

        :raises:    LoadedException: Forwards the exception that was raised
                    during execution, along with the input that caused it.
        """
        try:
            idx = 0
            async for interaction in input:
                idx += 1
                await interaction.perform(self.channel)
        except Exception as ex:
            raise LoadedException(ex, lambda: input[:idx]) from ex
        finally:
            ValueMeanProfiler("input_len", samples=100)(idx)
            CountProfiler("total_interactions")(idx)
