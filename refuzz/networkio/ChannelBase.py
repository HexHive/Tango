from . import debug

from   abc         import ABC, abstractmethod
from   common      import (ChannelBrokenException,
                          ChannelSetupException)
from   contextlib  import contextmanager
from   typing      import ByteString
from   dataclasses import dataclass
from   subprocess  import Popen

class ChannelBase(ABC):
    def __init__(self, pobj: Popen, timescale: float, **kwargs):
        self._pobj = pobj
        self._timescale = timescale
        self._sockfd = -1

    @abstractmethod
    def send(self, data: ByteString) -> int:
        pass

    @abstractmethod
    def receive(self) -> ByteString:
        pass

    @abstractmethod
    def close(self):
        pass

    @property
    def sockfd(self):
        return self._sockfd

@dataclass
class ChannelFactoryBase(ABC):
    """
    This class describes a channel's communication parameters.
    """

    timescale: float

    @abstractmethod
    def create(self) -> ChannelBase:
        pass

@dataclass
class TransportChannelFactory(ChannelFactoryBase):
    endpoint: str
    port: int