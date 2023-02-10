from   abc         import ABC, abstractmethod
from   typing      import ByteString
from   dataclasses import dataclass
from   common      import async_property

class ChannelBase(ABC):
    def __init__(self, timescale: float):
        self._timescale = timescale

    @abstractmethod
    async def send(self, data: ByteString) -> int:
        pass

    @abstractmethod
    async def receive(self) -> ByteString:
        pass

    @abstractmethod
    def close(self):
        pass

    @async_property
    async def timescale(self):
        return self._timescale

@dataclass
class ChannelFactoryBase(ABC):
    """
    This class describes a channel's communication parameters and can be used to
    instantiate a new channel.
    """
    timescale: float

    @abstractmethod
    def create(self, *args, **kwargs) -> ChannelBase:
        pass
