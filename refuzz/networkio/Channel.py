from   abc         import ABC, abstractmethod
from   common      import (ChannelBrokenException,
                          ChannelSetupException)
from   contextlib  import contextmanager
from   typing      import ByteString, Callable
from   dataclasses import dataclass
from   subprocess  import Popen
from   ptrace.debugger import (PtraceDebugger,
                               PtraceProcess,
                               ProcessExit,
                               ProcessSignal,
                               NewProcessEvent,
                               ProcessExecution)
from ptrace.func_call import FunctionCallOptions
from ptrace.syscall   import PtraceSyscall, SOCKET_SYSCALL_NAMES
from ptrace.tools import signal_to_exitcode
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import socket, select
import struct
import signal
import select
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
        self._proc = self._debugger.addProcess(self._pobj.pid, is_attached=True)
        self._proc.syscall_state.ignore_callback = self._ignore_callback

    def _ignore_callback(self, syscall):
        return syscall.name not in self._syscall_whitelist

    def _monitor_syscalls(self,
                       monitor_target: Callable,
                       ignore_callback: Callable[[PtraceSyscall], bool],
                       break_callback: Callable[..., bool],
                       syscall_callback: Callable[[PtraceProcess, PtraceSyscall], None],
                       break_on_entry: bool = False,
                       resume_process: bool = True,
                       **kwargs):
        with ThreadPoolExecutor() as executor:
            if not self._proc.is_stopped:
                self._proc.kill(signal.SIGSTOP)
                while True:
                    event = self._proc.waitEvent()
                    if event.signum in (signal.SIGSTOP, signal.SIGTRAP):
                        break

            ## Break on next syscall
            self._proc.syscall_state.ignore_callback = ignore_callback
            self._proc.syscall()

            ## Execute monitor target
            if monitor_target:
                future = executor.submit(monitor_target)

            ## Listen for and process syscalls
            while True:
                if not self._debugger:
                    raise ChannelBrokenException("Process was terminated while waiting for syscalls")

                try:
                    event = self._debugger.waitSyscall()
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
                    # TODO monitor child for syscalls as well? may be needed for multi-thread or multi-process targets
                    event.process.parent.syscall()
                    continue
                except ProcessExecution as event:
                    event.process.syscall()
                    continue

                # Process syscall enter or exit
                state = event.process.syscall_state
                syscall = state.event(self._syscall_options)
                # ensure that the syscall has finished successfully before callback
                if syscall and syscall.result != -1 and \
                        (break_on_entry or syscall.result is not None):
                    syscall_callback(event.process, syscall, **kwargs)

                if break_callback():
                    break

                # Break on next syscall
                event.process.syscall()

            ## Resume process and remove syscall breakpoints
            if resume_process:
                self._proc.cont()

            ## Return the target's result
            if monitor_target:
                return future.result()

    def __del__(self):
        if self._debugger and self._proc.is_attached:
            self._proc.terminate(wait_exit=True)

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

    class ListenerSocketState:
        SOCKET_UNBOUND = 1
        SOCKET_BOUND = 2
        SOCKET_LISTENING = 3

        def __init__(self):
            self._state = self.SOCKET_UNBOUND

        def event(self, process, syscall):
            assert (syscall.result != -1), f"Syscall failed, {errno.error_code(-syscall.result)}"
            if self._state == self.SOCKET_UNBOUND and syscall.name == 'bind':
                sockaddr = syscall.arguments[1].value
                # FIXME apparently, family is 2 bytes, not 4?
                self._sa_family,  = struct.unpack('@H', process.readBytes(sockaddr, 2))
                assert (self._sa_family  == socket.AF_INET), "Only IPv4 is supported."
                self._sin_port, = struct.unpack('!H', process.readBytes(sockaddr + 2, 2))
                self._sin_addr = socket.inet_ntop(self._sa_family, process.readBytes(sockaddr + 4, 4))
                self._state = self.SOCKET_BOUND
            elif self._state == self.SOCKET_BOUND and syscall.name == 'listen':
                self._state = self.SOCKET_LISTENING
                return (self._sa_family, self._sin_addr, self._sin_port)

    def __init__(self, pobj: Popen, tx_callback: Callable, rx_callback: Callable,
                 endpoint: str, port: int, timescale: float):
        super().__init__(pobj, tx_callback, rx_callback, timescale)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect((endpoint, port))

    def connect(self, address: tuple):
        def syscall_callback_listen(process, syscall, fds, listeners):
            if syscall.name == 'socket':
                domain = syscall.arguments[0].value
                typ = syscall.arguments[1].value
                if domain != socket.AF_INET or typ != socket.SOCK_STREAM:
                    return
                fd = syscall.result
                fds[fd] = self.ListenerSocketState()
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
        ignore_callback = lambda x: x.name not in ('socket', 'bind', 'listen')
        break_callback = lambda: any(x[2] == address[1] for x in listeners.values())

        self._monitor_syscalls(None, ignore_callback, break_callback, syscall_callback_listen, fds={}, listeners=listeners)

        listenfd = next(x[0] for x in listeners.items() if x[1][2] == address[1])

        ## Now that we've verified the server is listening, we can connect
        ## Listen for a call to accept() to get the connected socket fd
        sockfd = -1
        ignore_callback = lambda x: x.name not in ('accept', 'accept4')
        break_callback = lambda: sockfd > 0

        def syscall_callback_accept(process, syscall, listenfd):
            assert (syscall.name in ('accept', 'accept4')), "Received non-matching syscall."
            if syscall.arguments[0].value != listenfd:
                return
            nonlocal sockfd
            sockfd = syscall.result

        monitor_target = partial(self._socket.connect, address)
        self._monitor_syscalls(monitor_target, ignore_callback, break_callback, syscall_callback_accept, listenfd=listenfd)
        self._sockfd = sockfd

    def send(self, data: ByteString) -> int:
        sent = 0
        while sent < len(data):
            sent += self._send_sync(data[sent:])

        server_waiting = False
        # TODO add support for epoll?
        ignore_callback = lambda x: x.name not in ('read', 'recv', 'recvfrom', 'recvmsg', 'poll', 'ppoll', 'select')
        break_callback = lambda: server_waiting
        def syscall_callback_read(process, syscall):
            if syscall.name in ('poll', 'ppoll'):
                nfds = syscall.arguments[1].value
                pollfds = syscall.arguments[0].value
                fmt = '@ihh'
                size = struct.calcsize(fmt)
                for i in range(nfds):
                    fd, events, revents = struct.unpack(fmt, process.readBytes(pollfds + i * size, size))
                    if fd == self._sockfd and (events & select.POLLIN) != 0:
                        break
            elif syscall.name == 'select':
                nfds = syscall.arguments[0].value
                if nfds <= self._sockfd:
                    return
                readfds = syscall.arguments[1].value
                fmt = '@l'
                size = struct.calcsize(fmt)
                l_idx = self._sockfd // (size * 8)
                b_idx = self._sockfd % (size * 8)
                fd_set, = struct.unpack(fmt, process.readBytes(readfds + l_idx * size, size))
                if fd_set & (1 << b_idx) == 0:
                    return
            elif syscall.arguments[0].value != self._sockfd:
                return
            nonlocal server_waiting
            server_waiting = True
        self._monitor_syscalls(None, ignore_callback, break_callback, syscall_callback_read, break_on_entry=True)

        return sent

    def _send_sync(self, data: ByteString) -> int:
        server_received = 0
        client_sent = 0
        barrier = threading.Barrier(2)

        ignore_callback = lambda x: x.name not in ('read', 'recv', 'recvfrom', 'recvmsg')
        def break_callback():
            if not barrier.broken:
                barrier.wait()
                barrier.abort()
            return client_sent > 0 and server_received >= client_sent

        def send_monitor(data):
            nonlocal client_sent
            ret = self._socket.send(data)
            if ret == 0:
                raise ChannelBrokenException("Failed to send any data")
            client_sent = ret
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                # occurs sometimes when the barrier is broken while wait() has yet to finish
                pass
            return ret

        def syscall_callback_read(process, syscall):
            if syscall.arguments[0].value != self._sockfd:
                return
            nonlocal server_received
            if syscall.result == 0:
                raise ChannelBrokenException("Server failed to read data off socket")
            server_received += syscall.result

        monitor_target = partial(send_monitor, data)
        ret = self._monitor_syscalls(monitor_target, ignore_callback, break_callback, syscall_callback_read, resume_process=False)

        return ret

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

    def __del__(self):
        self._socket.close()
        super(TCPChannel, self).__del__()

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