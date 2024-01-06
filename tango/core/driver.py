from . import debug, info, warning, critical
from tango.core.input import AbstractInput
from tango.core.profiler import (FrequencyProfiler, EventProfiler,
                       ValueMeanProfiler, CountProfiler)
from tango.core.dataio    import AbstractChannelFactory, AbstractChannel
from tango.common import AsyncComponent, ComponentOwner, ComponentType
from tango.exceptions import LoadedException

from abc          import ABC, abstractmethod

__all__ = ['AbstractDriver', 'BaseDriver']

class AbstractDriver(AsyncComponent, ABC,
        component_type=ComponentType.driver):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if (target := cls.__dict__.get('execute_input')):
            wrapped = EventProfiler('execute_input')(
                FrequencyProfiler('execs', period=1)(target))
            setattr(cls, 'execute_input', wrapped)

    @abstractmethod
    async def relaunch(self):
        pass

    @abstractmethod
    async def execute_input(self, input: AbstractInput):
        pass

    @property
    @abstractmethod
    def channel(self) -> AbstractChannel:
        pass

class BaseDriver(AbstractDriver,
        capture_components={ComponentType.channel_factory}):
    def __init__(self, channel_factory: AbstractChannelFactory, **kwargs):
        super().__init__(**kwargs)
        self._factory = channel_factory

    async def initialize(self):
        debug("Done nothing")
        return await super().initialize()

    async def finalize(self, owner: ComponentOwner):
        debug("Done nothing")
        return await super().finalize(owner)

    async def execute_input(self, input: AbstractInput):
        """
        Executes the sequence of instructions specified by the input.

        :param      input:  An object derived from the AbstractInput abstract class.
        :type       input:  AbstractInput

        :raises:    LoadedException: Forwards the exception that was raised
                    during execution, along with the input that caused it.
        """
        try:
            idx = 0
            async for instruction in input:
                idx += 1
                await instruction.perform(self._channel)
        except Exception as ex:
            raise LoadedException(ex, lambda: input[:idx]) from ex
        finally:
            ValueMeanProfiler("input_len", samples=100)(idx)
            CountProfiler("total_instructions")(idx)
