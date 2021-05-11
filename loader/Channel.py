from   abc        import ABC, abstractmethod
from   common     import ChannelBrokenException
from   contextlib import contextmanager
from   typing     import ByteString, Callable
import socket, select

class ChannelBase(ABC):
    def __init__(self, tx_callback: Callable, rx_callback: Callable):
        self._tx_callback = tx_callback
        self._rx_callback = rx_callback

    @abstractmethod
    def send(self, data: ByteString) -> int:
        pass

    @abstractmethod
    def receive(self) -> ByteString:
        pass

class TCPChannel(ChannelBase):
    RECV_CHUNK_SIZE = 4096

    def __init__(self, tx_callback: Callable, rx_callback: Callable,
                 endpoint: str, port: int):
        super().__init__(tx_callback, rx_callback)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self, address: tuple):
        self._socket.connect(address)

    def __enter__(self):
        self._socket.connect((endpoint, port))
        return self

    def __exit__(self):
        self._socket.close()

    def send(self, data: ByteString) -> int:
        sent = 0
        while sent < len(data):
            ret = self._socket.send(data)
            if ret == 0:
                raise ChannelBrokenException("Failed to send any data")
            sent += ret

    def receive(self) -> ByteString:
        chunks = []
        while True:
            poll, _, _ = select.select([self._socket], [], [])
            if self._socket not in poll:
                return b''.join(chunks)

            ret = self._socket.recv(RECV_CHUNK_SIZE)
            if ret == b'' and len(chunks) == 0:
                raise ChannelBrokenException("recv returned 0, socket shutdown")

            chunks.append(ret)

class UDPChannel(ChannelBase):
    def __init__(self, tx_callback: Callable, rx_callback: Callable,
                 endpoint: str, port: int):
        super().__init__(tx_callback, rx_callback)
        raise NotImplemented()

@dataclass
class ChannelFactoryBase(ABC):
    """
    This class describes a channel's communication parameters.
    """

    tx_callback: Callable
    rx_callback: Callable

    @abstractmethod
    def create(self) -> ChannelBase:
        pass

@dataclass
class TransportChannelFactory(ChannelFactoryBase):
    endpoint: str
    port: int

@dataclass
class TCPChannelFactory(TransportChannelFactory):
    connect_timeout: float = 5.0 # seconds
    data_timeout: float = 5.0 # seconds
    def create(self) -> ChannelBase:
        return TCPChannel(self.tx_callback, self.rx_callback,
                          self.endpoint,    self.port)

@dataclass
class UDPChannelFactory(TransportChannelFactory):
    def create(self) -> ChannelBase:
        return UDPChannel(self.tx_callback, self.rx_callback,
                          self.endpoint,    self.port)

# TODO implement other channel factory data classes