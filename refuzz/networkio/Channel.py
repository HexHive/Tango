from   abc         import ABC, abstractmethod
from   common      import ChannelBrokenException,
                          ChannelSetupException
from   contextlib  import contextmanager
from   typing      import ByteString, Callable
from   dataclasses import dataclass
from   subprocess  import Popen
import psutil
import socket, select

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

    @property
    def sockfd(self):
        return self._sockfd

class TCPChannel(ChannelBase):
    RECV_CHUNK_SIZE = 4096
    CHAN_CREATE_RETRIES = 500
    CHAN_CREATE_WAIT = 0.001 # seconds

    def __init__(self, pobj: Popen, tx_callback: Callable, rx_callback: Callable,
                 endpoint: str, port: int, timescale: float):
        super().__init__(pobj, tx_callback, rx_callback, timescale)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect((endpoint, port))

    def connect(self, address: tuple):
        pid = self._pobj.pid
        proc = psutil.Process(pid)

        ## Search for listening socket in server process's connections
        retries = 0
        while True:
            try:
                # FIXME check for matching address too
                # TODO is this faster than attempting to connect?
                next(x for x in proc.connections('tcp')
                     if x.status == 'LISTEN' and
                        x.laddr.port == address[1])
                break
            except:
                retries += 1
                if retries == self.CHAN_CREATE_RETRIES:
                    raise ChannelSetupException("Failed to find listening socket")
                sleep(self.CHAN_CREATE_WAIT)

        ## Now that we've verified the server is listening, we can connect
        self._socket.connect(address)

        ## Once we're connected, we obtain the socket fd at the server process
        ## This is needed by the interaction synchronization code
        try:
            # FIXME check for matching address too
            conn = next(x for x in proc.connections('tcp')
                        if x.status == 'ESTABLISHED' and
                           x.raddr.port == self._socket.getsockname()[1])
        except StopIteration:
            raise ChannelSetupException("Failed to find established socket fd")
        except:
            raise

        self._sockfd = conn.fd

    def __del__(self):
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
            poll, _, _ = select.select([self._socket], [], [], 0)
            if self._socket not in poll:
                return b''.join(chunks)

            ret = self._socket.recv(RECV_CHUNK_SIZE)
            if ret == b'' and len(chunks) == 0:
                raise ChannelBrokenException("recv returned 0, socket shutdown")

            chunks.append(ret)

class UDPChannel(ChannelBase):
    def __init__(self, pobj: Popen, tx_callback: Callable, rx_callback: Callable,
                 endpoint: str, port: int, timescale: float):
        super().__init__(pobj, tx_callback, rx_callback, timescale)
        raise NotImplemented()

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

@dataclass
class TCPChannelFactory(TransportChannelFactory):
    connect_timeout: float = 5.0 # seconds
    data_timeout: float = 5.0 # seconds
    def create(self, pobj: Popen) -> ChannelBase:
        return TCPChannel(pobj,
                          self.tx_callback, self.rx_callback,
                          self.endpoint,    self.port,
                          timescale=self.timescale)

@dataclass
class UDPChannelFactory(TransportChannelFactory):
    def create(self, pobj: Popen) -> ChannelBase:
        return UDPChannel(pobj,
                          self.tx_callback, self.rx_callback,
                          self.endpoint,    self.port,
                          timescale=self.timescale)

# TODO implement other channel factory data classes