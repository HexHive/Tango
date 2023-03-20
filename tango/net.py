from . import debug, info

from tango.core import (AbstractChannel, FormatDescriptor, AbstractInstruction,
    TransmitInstruction, ReceiveInstruction, DelayInstruction,
    SerializedInputMeta)
from tango.ptrace.syscall import PtraceSyscall, SYSCALL_REGISTER
from tango.ptrace.debugger import PtraceProcess
from tango.unix import (PtraceChannel, PtraceForkChannel, PtraceChannelFactory,
    FileDescriptorChannel, FileDescriptorChannelFactory)
from tango.common import sync_to_async, GLOBAL_ASYNC_EXECUTOR
from tango.exceptions import ChannelBrokenException

from subprocess  import Popen
from dataclasses import dataclass
from pyroute2.netns import setns
from os import getpid
from typing import ByteString, Tuple, Iterable, Mapping, Any, Optional
from functools import cached_property
import errno
import struct
import select
import threading
import socket
import random
import time

from scapy.all import Ether, IP, TCP, UDP, Packet, PcapReader, PcapWriter, Raw

__all__ = [
    'NetworkFormatDescriptor', 'TransportFormatDescriptor', 'NetworkChannel',
    'NetworkChannelFactory', 'TransportChannelFactory', 'TCPChannelFactory',
    'TCPChannel', 'TCPForkChannelFactory', 'UDPChannelFactory', 'UDPChannel',
    'UDPForkChannelFactory', 'UDPForkChannel', 'PCAPInput'
]


class NetworkChannel(FileDescriptorChannel):
    def __init__(self, netns: str, **kwargs):
        super().__init__(**kwargs)
        self._netns = netns
        self._ctx = NetNSContext(nsname=self._netns)

    def nssocket(self, *args):
        """
        This is a wrapper for socket.socket() that creates the socket inside the
        specified network namespace.
        """
        with self._ctx:
            s = socket.socket(*args)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s

    ## Callbacks
    def _select_ignore_callback(self, syscall: PtraceSyscall) -> bool:
        return syscall.name not in ('recv', 'recvfrom', 'recvmsg') and \
            super()._select_ignore_callback(syscall)

    def _close_ignore_callback(self, syscall: PtraceSyscall) -> bool:
        return syscall.name != 'shutdown' and \
            super()._close_ignore_callback(syscall)

    @classmethod
    def _select_match_fds(cls, process: PtraceProcess, syscall: PtraceSyscall,
            fds: Iterable[int]) -> Optional[int]:
        matched_fd = super()._select_match_fds(process, syscall, fds)
        if matched_fd is None:
            match syscall.name:
                case 'recv' | 'recvfrom' | 'recvmsg':
                    if syscall.arguments[0].value not in fds:
                        return None
                    matched_fd = syscall.arguments[0].value
                case _:
                    return None
        return matched_fd

    def _close_syscall_exit_callback_internal(self,
            process: PtraceProcess, syscall: PtraceSyscall):
        if not super()._close_ignore_callback(syscall):
            super()._close_syscall_entry_callback_internal(process, syscall)
        else:
            if syscall.name == 'shutdown' and syscall.result == 0:
                raise ChannelBrokenException(
                    "Channel shutdown while waiting for server to read")

class NetNSContext (object):
    """
    This is a context manager that on entry assigns the process to an alternate
    network namespace (specified by name, filesystem path, or pid) and then
    re-assigns the process to its original network namespace on exit.
    """
    def __init__(self, nsname=None, nspath=None, nspid=None):
        self.mypath = self.get_ns_path(nspid=getpid())
        self.targetpath = self.get_ns_path(nspath, nsname=nsname, nspid=nspid)
        if not self.targetpath:
            raise ValueError('invalid namespace')
        self.myns = open(self.mypath)
        self.targetns = open(self.targetpath)

    def __enter__(self):
        # before entering a new namespace, we open a file descriptor in the
        # current namespace that we will use to restore our namespace on exit.
        setns(self.targetns.fileno(), 0)

    def __exit__(self, *args):
        setns(self.myns.fileno(), 0)

    def __del__(self):
        if hasattr(self, 'myns'):
            self.myns.close()
            self.targetns.close()

    @staticmethod
    def get_ns_path(nspath=None, nsname=None, nspid=None):
        """
        This is just a convenience function that will return the path to an
        appropriate namespace descriptor, give either a path, a network
        namespace name, or a pid.
        """
        if nsname:
            nspath = '/var/run/netns/%s' % nsname
        elif nspid:
            nspath = '/proc/%d/ns/net' % nspid

        return nspath

# kw_only=True is needed because typ is no longer a non-default attribute
@dataclass(kw_only=True, frozen=True)
class NetworkFormatDescriptor(FormatDescriptor):
    typ: str = 'pcap'

@dataclass(frozen=True)
class _TransportFormatDescriptor(FormatDescriptor):
    """
    This class is needed since `protocol` is a non-default attribute that will
    be specified at instantiation, but the `fmt` attribute is otherwise set
    to 'pcap', and it occurs before `protocol` in the reversed MRO.
    Using this class as the last class in the MRO places `protocol` right
    after the non-default `fmt`, satisfying the requirements for a dataclass.
    """
    protocol: str

@dataclass(frozen=True)
class TransportFormatDescriptor(NetworkFormatDescriptor, _TransportFormatDescriptor):
    def __get__(self, obj, owner):
        if obj is None:
            return self
        return getattr(obj, '_fmt')

    def __set__(self, obj, value):
        fmt = type(self)(protocol=value)
        object.__setattr__(obj, '_fmt', fmt)

@dataclass(kw_only=True, frozen=True)
class NetworkChannelFactory(FileDescriptorChannelFactory,
        capture_paths=['channel.endpoint']):
    endpoint: str

class PortDescriptor:
    def __get__(self, obj, owner):
        if obj is None:
            # no default value
            raise AttributeError
        return getattr(obj, '_port')

    def __set__(self, obj, value: 'str'):
        ival = int(value)
        object.__setattr__(obj, '_port', ival)

@dataclass(kw_only=True, frozen=True)
class TransportChannelFactory(NetworkChannelFactory,
        capture_paths=['channel.port']):
    port: PortDescriptor = PortDescriptor()
    protocol: str
    fmt: FormatDescriptor = TransportFormatDescriptor(protocol=None)

    def __post_init__(self):
        object.__setattr__(self, 'fmt', self.protocol) # implicit casting through the descriptor

    @property
    def fields(self) -> Mapping[str, Any]:
        d = super().fields
        return self.exclude_keys(d, 'protocol')

@dataclass(kw_only=True, frozen=True)
class TCPChannelFactory(TransportChannelFactory,
        capture_paths=['channel.connect_timeout']):
    connect_timeout: float = None # seconds

    protocol: str = "tcp"

    def create(self, pobj: Popen, netns: str) -> AbstractChannel:
        ch = TCPChannel(pobj=pobj, netns=netns, **self.fields)
        ch.connect((self.endpoint, self.port))
        return ch

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['channel'].get('type') == 'tcp'

class TCPChannel(NetworkChannel):
    RECV_CHUNK_SIZE = 4096

    def __init__(self, endpoint: str, port: int,
            connect_timeout: float, **kwargs):
        super().__init__(**kwargs)
        self._connect_timeout = connect_timeout * self._timescale if connect_timeout else None
        self._socket = None
        self._accept_process = None
        self._sockfd = -1
        self.setup((endpoint, port))

    def cb_socket_listening(self, process, syscall):
        pass

    def cb_socket_accepting(self, process, syscall):
        pass

    def cb_socket_accepted(self, process, syscall):
        pass

    def setup(self, address: tuple):
        ## Wait for a socket that is listening on the same port
        # FIXME check for matching address too
        self._setup_listeners = {}
        self._setup_address = address
        self._setup_accepting = False

        # wait until target sets up a listening socket
        self.monitor_syscalls(None, self._setup_ignore_callback,
            self._setup_break_callback, self._setup_syscall_callback,
            fds={}, listeners=self._setup_listeners, address=address,
            timeout=self._connect_timeout)

        # wait until socket begins accepting
        self.monitor_syscalls(None, self._setup_ignore_callback_accept,
            self._setup_break_callback_accept, self._setup_syscall_callback_accept,
            timeout=self._connect_timeout, break_on_entry=True,
            break_on_exit=False, listenfd=self._listenfd)

    def connect(self, address: tuple):
        self._socket = self.nssocket(socket.AF_INET, socket.SOCK_STREAM)
        ## Now that we've verified the server is listening, we can connect.
        ## Listen for a call to accept() to get the connected socket fd
        self._connect_address = address
        self._sockfd = -1
        self._refcounter.clear()
        self._accept_process, _ = self.monitor_syscalls(
            self._connect_monitor_target, self._connect_ignore_callback,
            self._connect_break_callback, self._connect_syscall_callback,
            listenfd=self._listenfd, timeout=self._connect_timeout)
        debug(f"Socket is now connected ({self._sockfd = })!")

        # wait for the next read, recv, select, or poll
        # or wait for the parent to fork, and trace child for these calls
        self.sync()
        assert self.synced

    # this is needed so that poll_sync is executed by GLOBAL_ASYNC_EXECUTOR
    # so that the ptraced child is accessed by the same tracer
    @sync_to_async(executor=GLOBAL_ASYNC_EXECUTOR)
    def a_sync(self):
        return self.sync()

    async def send(self, data: ByteString) -> int:
        # flush any data left in the receive buffer
        await self.receive()

        sent = 0
        while sent < len(data):
            sent += await self._send_sync(data[sent:])

        await self.a_sync()
        assert self.synced
        debug(f"Sent data to server: {data}")
        return sent

    @sync_to_async(executor=GLOBAL_ASYNC_EXECUTOR)
    def _send_sync(self, data: ByteString) -> int:
        self._send_server_received = 0
        self._send_client_sent = 0
        # Set up a barrier so that client_sent is ready when checking for break
        self._send_barrier = threading.Barrier(2)
        self._send_barrier_passed = False

        self._send_data = data
        _, ret = self.monitor_syscalls(self._send_send_monitor,
            self._send_ignore_callback, self._send_break_callback,
            self._send_syscall_callback, timeout=self._data_timeout)
        return ret

    @sync_to_async(executor=GLOBAL_ASYNC_EXECUTOR)
    def receive(self) -> ByteString:
        chunks = []
        while True:
            try:
                poll, _, _ = select.select([self._socket], [], [], 0)
                if self._socket not in poll:
                    data = b''.join(chunks)
                    if data:
                        debug(f"Received data from server: {data}")
                    return data
            except ValueError as ex:
                raise ChannelBrokenException(
                    "socket fd is negative, socket is closed") from ex

            try:
                ret = self._socket.recv(self.RECV_CHUNK_SIZE)
            except ConnectionResetError as ex:
                raise ChannelBrokenException("recv failed, connection reset") \
                    from ex
            if ret == b'' and len(chunks) == 0:
                raise ChannelBrokenException("recv returned 0, socket shutdown")
            elif ret == b'':
                data = b''.join(chunks)
                debug(f"Received data from server: {data}")
                return data

            chunks.append(ret)

    def close(self, **kwargs):
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        super().close(**kwargs)

    @classmethod
    def _select_match_fds(cls, process: PtraceProcess, syscall: PtraceSyscall,
            fds: Iterable[int]) -> Optional[int]:
        matched_fd = super()._select_match_fds(process, syscall, fds)
        if matched_fd is None:
            match syscall.name:
                case 'accept' | 'accept4':
                    if syscall.arguments[0].value not in fds:
                        return None
                    matched_fd = syscall.arguments[0].value
                case _:
                    return None
        return matched_fd

    ### Callbacks ###
    def _setup_syscall_callback(self, process, syscall, fds, listeners, address):
        if syscall.name == 'socket':
            domain = syscall.arguments[0].value
            typ = syscall.arguments[1].value
            if domain != socket.AF_INET or (typ & socket.SOCK_STREAM) == 0:
                return
            fd = syscall.result
            fds[fd] = ListenerSocketState()
        elif syscall.name in ('bind', 'listen'):
            fd = syscall.arguments[0].value
            if fd not in fds:
                return
            entry = syscall.result is None
            result = fds[fd].event(process, syscall, entry=entry)
            if result is not None:
                listeners[fd] = result
                candidates = [x[0] for x in listeners.items() if x[1][2] == address[1]]
                if candidates:
                    self._listenfd = candidates[0]
                    # At this point, the target is paused just after the listen syscall that
                    # matches our condition.
                    self.cb_socket_listening(process, syscall)

    def _setup_ignore_callback(self, syscall):
        return syscall.name not in ('socket', 'bind', 'listen')

    def _setup_break_callback(self):
        return any(x[2] == self._setup_address[1] for x in self._setup_listeners.values())

    def _setup_syscall_callback_accept(self, process, syscall, listenfd):
        assert syscall.result is None # only break_on_entry=True
        try:
            if self._select_match_fds(process, syscall, (listenfd,)) != listenfd:
                return
            self._setup_accepting = True
            self.cb_socket_accepting(process, syscall)
        finally:
            process.syscall()
            return True

    def _setup_ignore_callback_accept(self, syscall):
        return syscall.name not in ('accept', 'accept4', 'poll', 'ppoll',
            'select', 'pselect6')

    def _setup_break_callback_accept(self):
        return self._setup_accepting

    def _connect_syscall_callback(self, process, syscall, listenfd):
        if syscall.name in ('accept', 'accept4') \
                and syscall.arguments[0].value == listenfd \
                and syscall.result >= 0:
            self._sockfd = syscall.result
            self._refcounter[self._sockfd] = 1
            self.cb_socket_accepted(process, syscall)

    def _connect_ignore_callback(self, syscall):
        return syscall.name not in ('accept', 'accept4')

    def _connect_break_callback(self):
        return len(self._refcounter) > 0

    def _connect_monitor_target(self):
        return self._socket.connect(self._connect_address)

    def _send_syscall_callback(self, process, syscall):
        if syscall.name in ('read', 'recv', 'recvfrom', 'recvmsg') \
                and syscall.arguments[0].value in self._refcounter:
            if syscall.result <= 0:
                raise ChannelBrokenException(
                    "Server failed to read data off socket")
            self._send_server_received += syscall.result

    def _send_ignore_callback(self, syscall):
        return syscall.name not in ('read', 'recv', 'recvfrom', 'recvmsg')

    def _send_break_callback(self):
        if not self._send_barrier_passed:
            try:
                self._send_barrier.wait()
            except threading.BrokenBarrierError:
                raise ChannelBrokenException("Barrier broke while waiting")
            else:
                self._send_barrier_passed = True
        debug(f"{self._send_client_sent=}; {self._send_server_received=}")
        if self._send_client_sent == 0 or \
                self._send_server_received > self._send_client_sent:
            raise ChannelBrokenException(
                "Client sent no bytes, or server received too many bytes!")
        return self._send_server_received == self._send_client_sent

    def _send_send_monitor(self):
        try:
            ret = self._socket.send(self._send_data)
            if ret == 0:
                raise ChannelBrokenException("Failed to send any data")
        except Exception:
            self._send_barrier.abort()
            raise
        self._send_client_sent = ret
        self._send_barrier.wait()
        return ret

class ListenerSocketState:
    SOCKET_UNBOUND = 1
    SOCKET_BOUND = 2
    SOCKET_LISTENING = 3

    def __init__(self):
        self._state = self.SOCKET_UNBOUND

    def event(self, process, syscall, entry=False):
        assert (syscall.result != -1), f"Syscall failed," \
            f" {errno.error_code(-syscall.result)}"
        if self._state == self.SOCKET_UNBOUND and syscall.name == 'bind':
            sockaddr = syscall.arguments[1].value
            # FIXME apparently, family is 2 bytes, not 4?
            # maybe use the parser that comes with python-ptrace
            _sa_family,  = struct.unpack('@H', process.readBytes(sockaddr, 2))
            assert (_sa_family  == socket.AF_INET), "Only IPv4 is supported."
            _sin_port, = struct.unpack('!H', process.readBytes(sockaddr + 2, 2))
            _sin_addr = socket.inet_ntop(_sa_family, process.readBytes(
                sockaddr + 4, 4))
            if not entry:
                self._state = self.SOCKET_BOUND
                self._sa_family = _sa_family
                self._sin_port = _sin_port
                self._sin_addr = _sin_addr
                debug(f"Socket bound to port {self._sin_port}")
        elif self._state == self.SOCKET_BOUND and syscall.name == 'listen':
            if not entry:
                self._state = self.SOCKET_LISTENING
                debug(f"Server listening on port {self._sin_port}")
            return (self._sa_family, self._sin_addr, self._sin_port)


@dataclass(kw_only=True, frozen=True)
class TCPForkChannelFactory(TCPChannelFactory,
        capture_paths=['channel.fork_before_accept']):
    fork_before_accept: bool = True

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['driver'].get('forkserver')

    def create(self, pobj: Popen, netns: str) -> AbstractChannel:
        object.__setattr__(self, '_pobj', pobj)
        object.__setattr__(self, '_netns', netns)
        ch = self.forkchannel
        ch.connect((self.endpoint, self.port))
        return ch

    @cached_property
    def forkchannel(self):
        if self.fork_before_accept:
            return TCPForkBeforeAcceptChannel(pobj=self._pobj,
                          netns=self._netns, **self.fields)
        else:
            return TCPForkAfterListenChannel(pobj=self._pobj,
                          netns=self._netns, **self.fields)

    @property
    def fields(self) -> Mapping[str, Any]:
        d = super().fields
        return self.exclude_keys(d, 'fork_before_accept')

class TCPForkAfterListenChannel(TCPChannel, PtraceForkChannel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cb_socket_listening(self, process, syscall):
        self._invoke_forkserver(process)

class TCPForkBeforeAcceptChannel(TCPChannel, PtraceForkChannel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cb_socket_accepting(self, process, syscall):
        # The syscall instruction cannot be "cancelled". Instead we replace it
        # by a call to an invalid syscall so the kernel returns immediately, and
        # the forkserver is later invoked properly. Otherwise, the forkerser
        # process itself will accept the next connect request.
        self._syscall_num = syscall.syscall
        process.setreg(SYSCALL_REGISTER, -1)
        # we also change the name so that it is ignored by any unintended
        # listener
        syscall.name = "FORKSERVER"
        syscall.syscall = -1
        self._invoke_forkserver(process)

    def _invoke_forkserver(self, process):
        address = process.getInstrPointer() - 2
        self._inject_forkserver(process, address)

    def _cleanup_forkserver(self, process):
        # WARN the use of SYSCALL_REGISTER sets orig_rax, which differs from
        # 'rax' in this context.
        with process.regsctx():
            process.setreg('rax', self._syscall_num)
            super()._cleanup_forkserver(process)


@dataclass(kw_only=True, frozen=True)
class UDPChannelFactory(TransportChannelFactory,
        capture_paths=['channel.connect_timeout']):
    connect_timeout: float = None # seconds

    protocol: str = "udp"

    def create(self, pobj: Popen, netns: str) -> AbstractChannel:
        ch = UDPChannel(pobj=pobj, netns=netns, **self.fields)
        ch.connect((self.endpoint, self.port))
        return ch

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['channel'].get('type') == 'udp'

class UDPChannel(NetworkChannel):
    MAX_DATAGRAM_SIZE = 65507

    def __init__(self, endpoint: str, port: int,
                    connect_timeout: float, **kwargs):
        super().__init__(**kwargs)
        self._connect_timeout = connect_timeout * self._timescale if connect_timeout else None
        self._socket = None
        self._bind_process = None
        self._sockconnected = False
        self._sockfd = -1
        self.setup((endpoint, port))

    def cb_socket_binding(self, process, syscall):
        pass

    def cb_socket_bound(self, process, syscall):
        pass

    def setup(self, address: tuple):
        ## Wait for a socket that is listening on the same port
        # FIXME check for matching address too
        self._setup_binds = {}
        self._setup_address = address

        self.monitor_syscalls(None,
            self._setup_ignore_callback,
            self._setup_break_callback, self._setup_syscall_callback,
            timeout=self._connect_timeout, break_on_entry=True,
            break_on_exit=True, fds={}, binds=self._setup_binds, address=address)

    def connect(self, address: tuple):
        self._socket = self.nssocket(socket.AF_INET, socket.SOCK_DGRAM)
        self._refcounter.clear()
        self._sockconnected = False
        self._connect_address = address
        self._bind_process, _ = self.monitor_syscalls(
            self._connect_monitor_target, self._connect_ignore_callback,
            self._connect_break_callback, self._connect_syscall_callback,
            timeout=self._connect_timeout)

        debug(f"Socket is now connected ({self._sockfd = })!")

        # wait for the next read, recv, select, or poll
        # or wait for the parent to fork, and trace child for these calls
        self.sync()
        assert self.synced

    # this is needed so that poll_sync is executed by GLOBAL_ASYNC_EXECUTOR
    # so that the ptraced child is accessed by the same tracer
    @sync_to_async(executor=GLOBAL_ASYNC_EXECUTOR)
    def a_sync(self):
        return self.sync()

    async def send(self, data: ByteString) -> int:
        # flush any data left in the receive buffer
        while await self.receive():
            pass

        if len(data):
            sent = await self._send_sync(data)
            await self.a_sync()
            assert self.synced
        else:
            sent = 0
        debug(f"Sent data to server: {data[:sent]}")
        return sent

    @sync_to_async(executor=GLOBAL_ASYNC_EXECUTOR)
    def _send_sync(self, data: ByteString) -> int:
        self._send_server_received = 0
        self._send_client_sent = 0
        ## Set up a barrier so that client_sent is ready when checking for break condition
        self._send_barrier = threading.Barrier(2)
        self._send_barrier_passed = False

        self._send_data = data
        _, ret = self.monitor_syscalls(self._send_send_monitor,
            self._send_ignore_callback, self._send_break_callback,
            self._send_syscall_callback, timeout=self._data_timeout)
        return ret

    @sync_to_async(executor=GLOBAL_ASYNC_EXECUTOR)
    def receive(self) -> ByteString:
        try:
            poll, _, _ = select.select([self._socket], [], [], 0)
            if self._socket not in poll:
                return b''
        except ValueError:
            raise ChannelBrokenException("socket fd is negative, socket is closed")

        try:
            data = self._socket.recv(self.MAX_DATAGRAM_SIZE)
        except ConnectionResetError as ex:
            raise ChannelBrokenException("recv failed, connection reset") \
                from ex
        if data:
            debug(f"Received data from server: {data}")
        return data

    def close(self, **kwargs):
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        super().close(**kwargs)

    ### Callbacks ###
    def _setup_syscall_callback(self, process, syscall, fds, binds, address):
        is_entry = syscall.result is None
        try:
            if syscall.name == 'socket' and not is_entry:
                domain = syscall.arguments[0].value
                typ = syscall.arguments[1].value
                if domain != socket.AF_INET or (typ & socket.SOCK_DGRAM) == 0:
                    return
                fd = syscall.result
                fds[fd] = UDPSocketState()
            elif syscall.name == 'bind':
                fd = syscall.arguments[0].value
                if fd not in fds:
                    return
                # if syscall_entry, we'll do it again later
                result = fds[fd].event(process, syscall, entry=is_entry)
                if result is not None:
                    binds[fd] = result
                    candidates = [x[0] for x in binds.items() if x[1][2] == address[1]]
                    if candidates:
                        self._sockfd = candidates[0]
                        self.cb_socket_binding(process, syscall)
        finally:
            if is_entry:
                process.syscall()
            return is_entry

    def _setup_ignore_callback(self, syscall):
        return syscall.name not in ('socket', 'bind')

    def _setup_break_callback(self):
        return any(x[2] == self._setup_address[1] for x in self._setup_binds.values())

    def _connect_syscall_callback(self, process, syscall):
        if syscall.name == 'bind' \
                and syscall.arguments[0].value == self._sockfd \
                and syscall.result == 0:
            self._refcounter[self._sockfd] = 1
            self._sockconnected = True
            self.cb_socket_bound(process, syscall)

    def _connect_ignore_callback(self, syscall):
        return syscall.name != 'bind'

    def _connect_break_callback(self):
        return self._sockconnected

    def _connect_monitor_target(self):
        return self._socket.connect(self._connect_address)

    def _send_syscall_callback(self, process, syscall):
        if syscall.name in ('read', 'recv', 'recvfrom', 'recvmsg') \
                and syscall.arguments[0].value in self._refcounter:
            if syscall.result <= 0:
                raise ChannelBrokenException(
                    "Server failed to read data off socket")
            self._send_server_received = syscall.result

    def _send_ignore_callback(self, syscall):
        return syscall.name not in ('read', 'recv', 'recvfrom', 'recvmsg')

    def _send_break_callback(self):
        if not self._send_barrier_passed:
            try:
                self._send_barrier.wait()
            except threading.BrokenBarrierError:
                raise ChannelBrokenException("Barrier broke while waiting")
            else:
                self._send_barrier_passed = True
        debug(f"{self._send_client_sent=}; {self._send_server_received=}")
        if self._send_client_sent == 0 or \
                self._send_server_received > self._send_client_sent:
            raise ChannelBrokenException(
                "Client sent no bytes, or server received too many bytes!")
        return self._send_server_received > 0

    def _send_send_monitor(self):
        try:
            ret = self._socket.send(self._send_data)
            if ret == 0:
                raise ChannelBrokenException("Failed to send any data")
        except Exception:
            self._send_barrier.abort()
            raise
        self._send_client_sent = ret
        self._send_barrier.wait()
        return ret

class UDPSocketState:
    SOCKET_UNBOUND = 1
    SOCKET_BOUND = 2

    def __init__(self):
        self._state = self.SOCKET_UNBOUND

    def event(self, process, syscall, entry=False):
        assert (syscall.result != -1), f"Syscall failed, {errno.error_code(-syscall.result)}"
        if self._state == self.SOCKET_UNBOUND and syscall.name == 'bind':
            sockaddr = syscall.arguments[1].value
            # FIXME apparently, family is 2 bytes, not 4?
            # maybe use the parser that comes with python-ptrace
            _sa_family,  = struct.unpack('@H', process.readBytes(sockaddr, 2))
            assert (_sa_family  == socket.AF_INET), "Only IPv4 is supported."
            _sin_port, = struct.unpack('!H', process.readBytes(sockaddr + 2, 2))
            _sin_addr = socket.inet_ntop(_sa_family, process.readBytes(
                sockaddr + 4, 4))
            if not entry:
                self._state = self.SOCKET_BOUND
                self._sa_family = _sa_family
                self._sin_port = _sin_port
                self._sin_addr = _sin_addr
                debug(f"Socket bound to port {self._sin_port}")
            return (_sa_family, _sin_addr, _sin_port)


@dataclass(kw_only=True, frozen=True)
class UDPForkChannelFactory(UDPChannelFactory,
        capture_paths=['channel.fork_before_bind']):
    fork_before_bind: bool = False

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['driver'].get('forkserver')

    def create(self, pobj: Popen, netns: str) -> AbstractChannel:
        object.__setattr__(self, '_pobj', pobj)
        object.__setattr__(self, '_netns', netns)
        ch = self.forkchannel
        ch.connect((self.endpoint, self.port))
        return ch

    @cached_property
    def forkchannel(self):
        if self.fork_before_bind:
            return UDPForkBeforeBindChannel(pobj=self._pobj,
                              netns=self._netns, **self.fields)
        else:
            return UDPForkChannel(pobj=self._pobj,
                              netns=self._netns, **self.fields)

    @property
    def fields(self) -> Mapping[str, Any]:
        d = super().fields
        return self.exclude_keys(d, 'fork_before_bind')

class UDPForkChannel(UDPChannel, PtraceForkChannel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cb_socket_bound(self, process, syscall):
        self._invoke_forkserver(process)

    def connect(self, address: tuple):
        if not self._sockconnected:
            super().connect(address)
        else:
            self._socket = self.nssocket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.connect(address)
            self.sync()

class UDPForkBeforeBindChannel(UDPChannel, PtraceForkChannel):
    # Forking just before a bind will result in the same socket being bound in
    # multiple child processes (because they all copy a reference of the
    # parent's sockfd), and this is not allowed behavior by Linux's TCP/IP
    # stack. It will result in EINVAL (errno 22: Invalid argument)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cb_socket_binding(self, process, syscall):
        # The syscall instruction cannot be "cancelled". Instead we replace it
        # by a call to an invalid syscall so the kernel returns immediately, and
        # the forkserver is later invoked properly. Otherwise, the forkerser
        # process itself will accept the next connect request.
        self._syscall_num = syscall.syscall
        process.setreg(SYSCALL_REGISTER, -1)
        # we also change the name so that it is ignored by any unintended
        # listener
        syscall.name = "FORKSERVER"
        syscall.syscall = -1
        self._invoke_forkserver(process)

    def _invoke_forkserver(self, process):
        address = process.getInstrPointer() - 2
        self._inject_forkserver(process, address)

    def _cleanup_forkserver(self, process):
        # WARN the use of SYSCALL_REGISTER sets orig_rax, which differs from
        # 'rax' in this context.
        with process.regsctx():
            process.setreg('rax', self._syscall_num)
            super()._cleanup_forkserver(process)

class PCAPInput(metaclass=SerializedInputMeta, typ='pcap'):
    LAYER_SOURCE = {
        Ether: "src",
        IP: "src",
        TCP: "sport",
        UDP: "sport"
    }

    LAYER_DESTINATION = {
        Ether: "dst",
        IP: "dst",
        TCP: "dport",
        UDP: "dport"
    }

    DELAY_THRESHOLD = 1.0

    @classmethod
    def _try_identify_endpoints(cls, packet: Packet) -> Tuple:
        sender = []
        receiver = []
        for layer, src in cls.LAYER_SOURCE.items():
            if layer in packet:
                sender.append(getattr(packet.getlayer(layer), src))
        for layer, dst in cls.LAYER_DESTINATION.items():
            if layer in packet:
                receiver.append(getattr(packet.getlayer(layer), dst))
        if not sender or not receiver:
            raise RuntimeError("Could not identify endpoints in packet")
        return (tuple(sender), tuple(receiver))

    def loadi(self) -> Iterable[AbstractInstruction]:
        if self._fmt.protocol in ("tcp", "udp"):
            layer = Raw
        else:
            layer = None
        plist = PcapReader(self._file).read_all()
        endpoints = []
        for p in plist:
            eps = self._try_identify_endpoints(p)
            for endpoint in eps:
                if endpoint not in endpoints:
                    endpoints.append(endpoint)
        if len(endpoints) != 2:
            raise RuntimeError(
                f"PCAP file has {len(endpoints)} endpoints (expected 2)"
            )

        if layer:
            plist = [p for p in plist if p.haslayer(layer)]

        tlast = plist[0].time
        for p in plist:
            # FIXME this operation is done previously, optimize
            src, dst = self._try_identify_endpoints(p)

            if layer:
                payload = bytes(p.getlayer(layer))
            else:
                payload = bytes(p)

            delay = p.time - tlast
            tlast = p.time
            if delay >= self.DELAY_THRESHOLD:
                yield DelayInstruction(float(delay))

            if src == endpoints[0]:
                instruction = TransmitInstruction(data=payload)
            else:
                instruction = ReceiveInstruction(data=payload)
            yield instruction

    def dumpi(self, itr: Iterable[AbstractInstruction], /):
        if self._fmt.protocol == "tcp":
            layer = TCP
            cli = random.randint(40000, 65534)
            srv = random.randint(cli + 1, 65535)
        elif self._fmt.protocol == "udp":
            layer = UDP
            cli = random.randint(40000, 65534)
            srv = random.randint(cli + 1, 65535)
        else:
            raise NotImplementedError

        cur_time = time.time()
        writer = PcapWriter(self._file)
        client_sent = False
        for instruction in itr:
            if isinstance(instruction, DelayInstruction):
                if instruction._time >= self.DELAY_THRESHOLD:
                    cur_time += instruction._time
                continue
            elif isinstance(instruction, TransmitInstruction):
                src, dst = cli, srv
                client_sent = True
            elif isinstance(instruction, ReceiveInstruction):
                if not client_sent:
                    continue
                src, dst = srv, cli
            p = Ether(src='aa:aa:aa:aa:aa:aa', dst='aa:aa:aa:aa:aa:aa') / IP() / \
                    layer(**{self.LAYER_SOURCE[layer]: src, self.LAYER_DESTINATION[layer]: dst}) / \
                        Raw(load=instruction._data)
            p.time = cur_time
            writer.write(p)

    def __repr__(self):
        return f"PCAPInput({self._file})"
