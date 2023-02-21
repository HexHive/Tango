from __future__ import annotations

from tango.common      import async_property, Configurable, ComponentType
from tango.core.profiler import FrequencyProfiler, EventProfiler, CountProfiler

from abc         import ABC, abstractmethod
from typing      import ByteString
from dataclasses import dataclass
from math        import isclose
from asyncio     import sleep

__all__ = [
    'FormatDescriptor', 'AbstractChannel', 'AbstractChannelFactory',
    'AbstractInstruction', 'TransmitInstruction', 'ReceiveInstruction',
    'DelayInstruction'
]


@dataclass(frozen=True)
class FormatDescriptor:
    """
    This is just a helper class to be used when passing around
    information describing the format of the serialized data.
    Additional fields could be added by inheritance, in case a
    format can be further specialized, depending on the channel
    being used.
    """
    typ: str

class AbstractChannel(ABC):
    def __init__(self, timescale: float):
        self._timescale = timescale

    # FIXME these type annotations are way out of date
    @abstractmethod
    async def send(self, data: ByteString) -> int:
        pass

    @abstractmethod
    async def receive(self) -> ByteString:
        pass

    @abstractmethod
    def close(self):
        pass

    @property
    def timescale(self):
        return self._timescale

class TimescaleDescriptor:
    def __init__(self, *, default: float=1.0):
        self._default = default

    def __get__(self, obj, owner):
        if obj is None:
            return self._default
        return getattr(obj, '_timescale')

    def __set__(self, obj, value: str):
        fval = float(value)
        setattr(obj, '_timescale', fval)

@dataclass
class AbstractChannelFactory(Configurable, ABC,
        component_type=ComponentType.channel_factory,
        capture_paths=['fuzzer.timescale']):
    """
    This class describes a channel's communication parameters and can be used to
    instantiate a new channel.
    """
    fmt: FormatDescriptor
    timescale: TimescaleDescriptor = TimescaleDescriptor()

    @abstractmethod
    def create(self, *args, **kwargs) -> AbstractChannel:
        pass

class AbstractInstruction(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if (target := cls.__dict__.get('perform')):
            wrapped = EventProfiler('perform_interaction')(
                FrequencyProfiler('interactions', period=1)(target))
            setattr(cls, 'perform', wrapped)

    @abstractmethod
    async def perform(self, channel: AbstractChannel):
        pass

    @abstractmethod
    def __eq__(self, other: AbstractInstruction):
        pass

class TransmitInstruction(AbstractInstruction):
    def __init__(self, data: ByteString):
        self._data = data

    async def perform(self, channel: AbstractChannel):
        sent = await channel.send(self._data)
        if sent < len(self._data):
            self._data = self._data[:sent]
        CountProfiler('bytes_sent')(sent)

    def __eq__(self, other: TransmitInstruction):
        return isinstance(other, TransmitInstruction) and self._data == other._data

    def __repr__(self):
        return f'<tx data="{self._data}">'

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__init__(self._data)
        return result

class ReceiveInstruction(AbstractInstruction):
    def __init__(self, size: int = 0, data: ByteString = None, expected: ByteString = None):
        self._size = size
        self._expected = expected
        self._data = data

    async def perform(self, channel: AbstractChannel):
        self._data = await channel.receive()
        # TODO verify size? verify data??

    def __repr__(self):
        return f'<rx data="{self._data}">'

    def __eq__(self, other: ReceiveInstruction):
        # FIXME make this depend on the expected data?
        return isinstance(other, ReceiveInstruction)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__init__(self._size, self._data, self._expected)
        return result

class DelayInstruction(AbstractInstruction):
    def __init__(self, time: float):
        self._time = float(time)
        # a delay longer than 5 seconds is probably useless, but we'll keep this
        # configurable
        self._maxdelay = 5

    async def perform(self, channel: AbstractChannel):
        await sleep(self._time * channel.timescale)

    def __eq__(self, other: DelayInstruction):
        return isinstance(other, DelayInstruction) and \
            isclose(self._time, other._time, rel_tol=0.01)

    def __repr__(self):
        return f'sleep({self._time})'

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__init__(self._time)
        return result