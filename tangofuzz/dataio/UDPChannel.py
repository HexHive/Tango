from . import debug, info

from typing import ByteString
from dataio import ChannelBase, PtraceChannel, TransportChannelFactory
from   common      import (ChannelBrokenException,
                          ChannelSetupException)
from subprocess import Popen
import errno
import struct
import socket
import select
import threading
from dataclasses import dataclass
from functools import partial
from ptrace import PtraceError
from common import sync_to_async, GLOBAL_ASYNC_EXECUTOR

@dataclass
class UDPChannelFactory(TransportChannelFactory):
    connect_timeout: float = None # seconds
    data_timeout: float = None # seconds

    protocol: str = "udp"

    def create(self, pobj: Popen, netns: str, *args, **kwargs) -> ChannelBase:
        ch = UDPChannel(pobj=pobj,
                          netns=netns,
                          endpoint=self.endpoint, port=self.port,
                          timescale=self.timescale,
                          connect_timeout=self.connect_timeout,
                          data_timeout=self.data_timeout)
        ch.connect((self.endpoint, self.port))
        return ch

class UDPChannel(PtraceChannel):
    MAX_DATAGRAM_SIZE = 65507

    def __init__(self, endpoint: str, port: int,
                    connect_timeout: float, data_timeout: float, **kwargs):
        super().__init__(**kwargs)
        self._connect_timeout = connect_timeout * self._timescale if connect_timeout else None
        self._data_timeout = data_timeout * self._timescale if data_timeout else None
        self._socket = None
        self._bind_process = None
        self._refcounter = 0
        self._sockconnected = False
        self._sockfd = -1
        self.setup((endpoint, port))

    def cb_socket_binding(self, process, syscall):
        pass

    def cb_socket_bound(self, process, syscall):
        pass

    def process_new(self, *args, **kwargs):
        super().process_new(*args, **kwargs)
        # FIXME self._socket could be not None but the socket may not yet be
        # accepted (see: connect())
        if self._socket is not None:
            self._refcounter += 1

    def setup(self, address: tuple):
        ## Wait for a socket that is listening on the same port
        # FIXME check for matching address too
        self._setup_binds = {}
        self._setup_address = address

        self.monitor_syscalls(None, \
            self._setup_ignore_callback, \
            self._setup_break_callback, self._setup_syscall_callback, \
            timeout=self._connect_timeout, break_on_entry=True, \
            fds={}, binds=self._setup_binds, address=address)

        del self._setup_binds
        del self._setup_address

    def connect(self, address: tuple):
        self._socket = self.nssocket(socket.AF_INET, socket.SOCK_DGRAM)
        self._refcounter = 0
        self._sockconnected = False
        self._connect_address = address
        self._bind_process, _ = self.monitor_syscalls( \
            self._connect_monitor_target, self._connect_ignore_callback, \
            self._connect_break_callback, self._connect_syscall_callback, \
            timeout=self._connect_timeout)
        del self._connect_address

        debug(f"Socket is now connected ({self._sockfd = })!")

        # wait for the next read, recv, select, or poll
        # or wait for the parent to fork, and trace child for these calls
        self._poll_sync()

    def _poll_sync(self):
        self._poll_server_waiting = False
        proc, _ = self.monitor_syscalls(None, self._poll_ignore_callback, \
            self._poll_break_callback, self._poll_syscall_callback, \
            break_on_entry=True, timeout=self._data_timeout)
        del self._poll_server_waiting

    @sync_to_async(executor=GLOBAL_ASYNC_EXECUTOR)
    def send(self, data: ByteString) -> int:
        if len(data):
            sent = self._send_sync(data)
            self._poll_sync()
        else:
            sent = 0
        debug(f"Sent data to server: {data[:sent]}")
        return sent

    def _send_sync(self, data: ByteString) -> int:
        self._send_server_received = 0
        self._send_client_sent = 0
        ## Set up a barrier so that client_sent is ready when checking for break condition
        self._send_barrier = threading.Barrier(2)

        self._send_data = data
        _, ret = self.monitor_syscalls(self._send_send_monitor, \
            self._send_ignore_callback, self._send_break_callback, \
            self._send_syscall_callback, timeout=self._data_timeout)

        del self._send_server_received
        del self._send_client_sent
        del self._send_barrier
        del self._send_data

        return ret

    @sync_to_async(executor=GLOBAL_ASYNC_EXECUTOR)
    def receive(self) -> ByteString:
        try:
            poll, _, _ = select.select([self._socket], [], [], 0)
            if self._socket not in poll:
                debug(f"Received nothing from the server!")
                return b''
        except ValueError:
            raise ChannelBrokenException("socket fd is negative, socket is closed")

        data = self._socket.recv(self.MAX_DATAGRAM_SIZE)
        if data == b'':
            raise ChannelBrokenException("recv returned 0, socket shutdown")
        else:
            debug(f"Received data from server: {data}")
            return data

    def close(self, **kwargs):
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        super().close(**kwargs)

    ### Callbacks ###
    def _setup_syscall_callback(self, process, syscall, fds, binds, address):
        try:
            if syscall.name == 'socket':
                domain = syscall.arguments[0].value
                typ = syscall.arguments[1].value
                if domain != socket.AF_INET or (typ & socket.SOCK_DGRAM) == 0:
                    return
                if syscall.result is None:
                    return
                fd = syscall.result
                fds[fd] = UDPSocketState()
            elif syscall.name == 'bind':
                fd = syscall.arguments[0].value
                if fd not in fds:
                    return
                entry = syscall.result is None # if syscall_entry, we'll do it again later
                result = fds[fd].event(process, syscall, entry=entry)
                if result is not None:
                    binds[fd] = result
                    candidates = [x[0] for x in binds.items() if x[1][2] == address[1]]
                    if candidates:
                        self._sockfd = candidates[0]
                        self.cb_socket_binding(process, syscall)
        finally:
            process.syscall()

    def _setup_ignore_callback(self, syscall):
        return syscall.name not in ('socket', 'bind')

    def _setup_break_callback(self):
        return any(x[2] == self._setup_address[1] for x in self._setup_binds.values())

    def _poll_syscall_callback(self, process, syscall):
        try:
            # poll, ppoll
            if syscall.name in ('poll', 'ppoll'):
                nfds = syscall.arguments[1].value
                pollfds = syscall.arguments[0].value
                fmt = '@ihh'
                size = struct.calcsize(fmt)
                for i in range(nfds):
                    fd, events, revents = struct.unpack(fmt, process.readBytes(pollfds + i * size, size))
                    if fd == self._sockfd and (events & select.POLLIN) != 0:
                        break
                else:
                    return
            # select
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
            # read, recv, recvfrom, recvmsg
            elif syscall.name in ('read', 'recv', 'recvfrom', 'recvmsg') and syscall.arguments[0].value == self._sockfd:
                pass
            # For the next two cases, since break_on_entry is True, the effect
            # only takes place on the second occurrence, when the syscall exits;
            # so we check if the result is None (entry) or not (exit)
            elif syscall.name in ('shutdown', 'close') and syscall.arguments[0].value == self._sockfd:
                if syscall.result is None:
                    return
                elif syscall.name == 'shutdown':
                    raise ChannelBrokenException("Channel closed while waiting for server to read")
                elif syscall.name == 'close':
                    self._refcounter -= 1
                    if self._refcounter == 0:
                        raise ChannelBrokenException("Channel closed while waiting for server to read")
                    return
            else:
                return
        finally:
            process.syscall()
        self._poll_server_waiting = True

    def _poll_ignore_callback(self, syscall):
        # TODO add support for epoll?
        return syscall.name not in ('read', 'recv', 'recvfrom', 'recvmsg', \
                                'poll', 'ppoll', 'select', 'close', 'shutdown')

    def _poll_break_callback(self):
        return self._poll_server_waiting

    def _connect_syscall_callback(self, process, syscall):
        if syscall.name in ('bind',) \
                and syscall.arguments[0].value == self._sockfd \
                and syscall.result == 0:
            self._refcounter = 1
            self._sockconnected = True
            self.cb_socket_bound(process, syscall)

    def _connect_ignore_callback(self, syscall):
        return syscall.name not in ('bind',)

    def _connect_break_callback(self):
        return self._sockconnected

    def _connect_monitor_target(self):
        return self._socket.connect(self._connect_address)

    def _send_syscall_callback(self, process, syscall):
        if syscall.name in ('read', 'recv', 'recvfrom', 'recvmsg') \
                and syscall.arguments[0].value == self._sockfd:
            if syscall.result == 0:
                raise ChannelBrokenException("Server failed to read data off socket")
            self._send_server_received = syscall.result
        elif syscall.name == 'shutdown' and syscall.arguments[0].value == self._sockfd:
            raise ChannelBrokenException("Channel closed while waiting for server to read")
        elif syscall.name == 'close' and syscall.arguments[0].value == self._sockfd:
            self._refcounter -= 1
            if self._refcounter == 0:
                raise ChannelBrokenException("Channel closed while waiting for server to read")

    def _send_ignore_callback(self, syscall):
        return syscall.name not in ('read', 'recv', 'recvfrom', 'recvmsg', \
                                    'close', 'shutdown')

    def _send_break_callback(self):
        if not self._send_barrier.broken:
            self._send_barrier.wait()
            self._send_barrier.abort()
        debug(f"{self._send_client_sent=}; {self._send_server_received=}")
        assert (self._send_client_sent > 0 \
                    and self._send_server_received <= self._send_client_sent), \
            "Client sent no bytes, or server received too many bytes!"
        return self._send_server_received > 0

    def _send_send_monitor(self):
        ret = self._socket.send(self._send_data)
        if ret == 0:
            raise ChannelBrokenException("Failed to send any data")
        self._send_client_sent = ret
        try:
            self._send_barrier.wait()
        except threading.BrokenBarrierError:
            # occurs sometimes when the barrier is broken while wait() has yet to finish
            # but it's benign
            pass
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
            _sin_addr = socket.inet_ntop(_sa_family, process.readBytes(sockaddr + 4, 4))
            if not entry:
                self._state = self.SOCKET_BOUND
                self._sa_family = _sa_family
                self._sin_port = _sin_port
                self._sin_addr = _sin_addr
                debug(f"Socket bound to port {self._sin_port}")
            return (_sa_family, _sin_addr, _sin_port)