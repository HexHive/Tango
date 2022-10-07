from   abc         import ABC, abstractmethod
from   typing      import ByteString
from   dataclasses import dataclass
from   common      import async_property

class ChannelBase(ABC):
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
    @abstractmethod
    async def timescale(self):
        pass

@dataclass
class ChannelFactoryBase(ABC):
    """
    This class describes a channel's communication parameters and can be used to
    instantiate a new channel.
    """

    @abstractmethod
    def create(self, *args, **kwargs) -> ChannelBase:
        pass
