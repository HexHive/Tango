from . import debug

from   abc         import ABC, abstractmethod
from   common      import (ChannelBrokenException,
                          ChannelSetupException)
from   contextlib  import contextmanager
from   typing      import ByteString, Callable
from   dataclasses import dataclass
from   subprocess  import Popen

class ChannelBase(ABC):
    def __init__(self, pobj: Popen, tx_callback: Callable, rx_callback: Callable,
            timescale: float):
        self._pobj = pobj
        self._tx_callback = tx_callback
        self._rx_callback = rx_callback
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

    tx_callback: Callable
    rx_callback: Callable
    timescale: float

    @abstractmethod
    def create(self) -> ChannelBase:
        # TODO set up the recv hook and tx_callback stuff
        pass

@dataclass
class TransportChannelFactory(ChannelFactoryBase):
    endpoint: str
    port: int