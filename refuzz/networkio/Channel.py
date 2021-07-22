from   abc         import ABC, abstractmethod
from   common      import ChannelBrokenException,
                          ChannelSetupException
from   contextlib  import contextmanager
from   typing      import ByteString, Callable
from   dataclasses import dataclass
from   subprocess  import Popen
from   ptrace.debugger import (PtraceDebugger,
                               ProcessExit,
                               ProcessSignal,
                               NewProcessEvent,
                               ProcessExecution)
from ptrace.func_call import FunctionCallOptions
from ptrace.syscall   import PtraceSyscall, SOCKET_SYSCALL_NAMES
import socket, select
import struct
import threading

SOCKET_SYSCALL_NAMES = SOCKET_SYSCALL_NAMES.union(('read', 'write'))

class ChannelBase(ABC):
    def __init__(self, pobj: Popen, tx_callback: Callable, rx_callback: Callable,
            timescale: float):
        self._pobj = pobj
        self._tx_callback = tx_callback
        self._rx_callback = rx_callback
        self._timescale = timescale
        self._sockfd = -1

        self._syscall_options = FunctionCallOptions(
            write_types=False,
            write_argname=False,
            string_max_length=300,
            replace_socketcall=True,
            write_address=False,
            max_array_count=20,
        )
        self._syscall_options.instr_pointer = False
        self._syscall_whitelist = SOCKET_SYSCALL_NAMES

        self._debugger = PtraceDebugger()
        self._proc = dbg.addProcess(self._pobj.pid, is_attached=True)
        self._proc.syscall_state.ignore_callback = self._ignore_callback

    def _ignore_callback(self, syscall):
        return syscall.name not in self._syscall_whitelist

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

    class ListenerSocketState:
        SOCKET_UNBOUND = 1
        SOCKET_BOUND = 2
        SOCKET_LISTENING = 3

        def __init__(self):
            self._state = SOCKET_UNBOUND

        def event(self, process, syscall):
            assert(syscall.result != -1, f"Syscall failed, {errno.error_code(-syscall.result)}")
            if self._state == SOCKET_UNBOUND and syscall.name == 'bind'
                sockaddr = syscall.arguments[1].value
                # FIXME apparently, family is 2 bytes, not 4?
                self._sa_family  = struct.unpack('=H', process.readBytes(sockaddr, 2))
                assert(self._sa_family  == socket.AF_INET, "Only IPv4 is supported.")
                self._sin_port = struct.unpack('>H', process.readBytes(sockaddr + 2, 2))
                self._sin_addr = socket.inet_ntop(self._sa_family, process.readBytes(sockaddr + 4, 4))
                self._state = SOCKET_BOUND
            else if self._state == SOCKET_BOUND and syscall.name == 'listen'
                self._state = SOCKET_LISTENING
                return (self._sa_family, self._sin_addr, self._sin_port)

    def __init__(self, pobj: Popen, tx_callback: Callable, rx_callback: Callable,
                 endpoint: str, port: int, timescale: float):
        super().__init__(pobj, tx_callback, rx_callback, timescale)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect((endpoint, port))

    def _wait_syscalls(self,
                       ignore_callback: Callable[[PtraceSyscall], bool],
                       break_callback: Callable[..., bool],
                       syscall_callback: Callable[[PtraceProcess, PtraceSyscall], None],
                       **kwargs) -> threading.Thread
        def waiter():
            pid = self._pobj.pid
            dbg = PtraceDebugger()
            # this assumes that the loader launched the target with PTRACE_TRACEME
            proc = dbg.addProcess(pid, is_attached=True)

            syscall_options = FunctionCallOptions(
                write_types=False,
                write_argname=False,
                string_max_length=300,
                replace_socketcall=True,
                write_address=False,
                max_array_count=20,
            )
            syscall_options.instr_pointer = False

            proc.syscall()
            proc.syscall_state.ignore_callback = ignore_callback

            while not break_callback()
                if not dbg
                    raise ChannelBrokenException("Process was terminated while waiting for syscalls")

                try:
                    event = dbg.waitSyscall()
                except ProcessExit as event:
                    exitcode = -256
                    if event.exitcode is not None:
                        exitcode = event.exitcode
                    raise ChannelBrokenException(f"Process exited with code {exitcode}")
                except ProcessSignal as event:
                    event.display()
                    event.process.syscall(event.signum)
                    exitcode = signal_to_exitcode(event.signum)
                    continue
                except NewProcessEvent as event:
                    continue
                except ProcessExecution as event:
                    continue

                # Process syscall enter or exit
                state = event.process.syscall_state
                syscall = state.event(syscall_options)
                # ensure that the syscall has finished successfully before callback
                if syscall and syscall.result != -1:
                    syscall_callback(event.process, syscall, **kwargs)

                # Break at next syscall
                event.process.syscall()

            # Resume process
            proc.cont()
        return threading.Thread(target=waiter)

    def connect(self, address: tuple):
        def syscall_callback_listen(process, syscall, fds, listeners):
            if syscall.name == 'socket':
                domain = syscall.arguments[0].value
                typ = syscall.arguments[1].value
                if domain != socket.AF_INET or typ != socket.SOCK_STREAM:
                    return
                fd = syscall.result
                fds[fd] = ListenerSocketState()
            else:
                fd = syscall.arguments[0].value
                if fd not in fds:
                    return
                result = fds[fd].event(process, syscall)
                if result is not None:
                    listeners[fd] = result


        ## Wait for a socket that is listening on the same port
        # FIXME check for matching address too
        listeners = {}
        ignore_callback = lambda x: x not in ('socket', 'bind', 'listen')
        break_callback = lambda: any(x[2] == address[1] for x in listeners.values())

        self._wait_syscalls(ignore_callback, break_callback, syscall_callback_listen, fds={}).join()

        listenfd = next(x[0] for x in listeners.items() if x[1][2] == address[1])

        ## Listen for a call to accept() to get the connected socket fd
        sockfd = -1
        ignore_callback = lambda x: x != 'accept'
        break_callback = lambda: sockfd > 0

        def syscall_callback_accept(process, syscall, listenfd):
            nonlocal sockfd
            fd = syscall.arguments[0].value
            if fd == listenfd:
                sockfd = syscall.result

        waiter = self._wait_syscalls(ignore_callback, break_callback, syscall_callback_accept, fds={})

        ## Now that we've verified the server is listening, we can connect
        self._socket.connect(address)

        ## Wait for the accept() syscall to go through and record sockfd
        waiter.join()
        self._sockfd = sockfd

        rasie ChannelSetupException(f"Obtained {sockfd=}")

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