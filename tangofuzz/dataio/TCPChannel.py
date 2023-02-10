from . import debug, info

from typing import ByteString
from dataio import (ChannelBase, PtraceChannel,
                    NetworkChannel, TransportChannelFactory)
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

@dataclass(kw_only=True)
class TCPChannelFactory(TransportChannelFactory):
    connect_timeout: float = None # seconds
    data_timeout: float = None # seconds

    protocol: str = "tcp"

    def create(self, pobj: Popen, netns: str, *args, **kwargs) -> ChannelBase:
        ch = TCPChannel(pobj=pobj,
                          netns=netns,
                          endpoint=self.endpoint, port=self.port,
                          timescale=self.timescale,
                          connect_timeout=self.connect_timeout,
                          data_timeout=self.data_timeout)
        ch.connect((self.endpoint, self.port))
        return ch

class TCPChannel(PtraceChannel, NetworkChannel):
    RECV_CHUNK_SIZE = 4096

    def __init__(self, endpoint: str, port: int,
                    connect_timeout: float, data_timeout: float, **kwargs):
        super().__init__(**kwargs)
        self._connect_timeout = connect_timeout * self._timescale if connect_timeout else None
        self._data_timeout = data_timeout * self._timescale if data_timeout else None
        self._socket = None
        self._accept_process = None
        self._refcounter = dict()
        self._sockfd = set()
        self.setup((endpoint, port))

    def cb_socket_listening(self, process, syscall):
        pass

    def cb_socket_accepting(self, process, syscall):
        pass

    def cb_socket_accepted(self, process, syscall):
        pass

    def process_new(self, *args, **kwargs):
        super().process_new(*args, **kwargs)
        # FIXME self._socket could be not None but the socket may not yet be
        # accepted (see: connect())
        if self._socket is not None:
            for fd in self._refcounter:
                self._refcounter[fd] += 1

    def setup(self, address: tuple):
        ## Wait for a socket that is listening on the same port
        # FIXME check for matching address too
        self._setup_listeners = {}
        self._setup_address = address
        self._setup_accepting = False

        self.monitor_syscalls(None, self._setup_ignore_callback, \
            self._setup_break_callback, self._setup_syscall_callback, \
            fds={}, listeners=self._setup_listeners, address=address, \
            timeout=self._connect_timeout)

        self.monitor_syscalls(None, self._setup_ignore_callback_accept, \
            self._setup_break_callback_accept, self._setup_syscall_callback_accept, \
            timeout=self._connect_timeout, break_on_entry=True, \
            listenfd=self._listenfd)

        del self._setup_listeners
        del self._setup_address
        del self._setup_accepting

    def connect(self, address: tuple):
        self._socket = self.nssocket(socket.AF_INET, socket.SOCK_STREAM)
        ## Now that we've verified the server is listening, we can connect
        ## Listen for a call to accept() to get the connected socket fd
        self._connect_address = address
        self._sockfd.clear()
        self._accept_process, _ = self.monitor_syscalls( \
            self._connect_monitor_target, self._connect_ignore_callback, \
            self._connect_break_callback, self._connect_syscall_callback, \
            listenfd=self._listenfd, timeout=self._connect_timeout)
        del self._connect_address
        debug(f"Socket is now connected ({self._sockfd = })!")

        # wait for the next read, recv, select, or poll
        # or wait for the parent to fork, and trace child for these calls
        self._poll_sync()

    def _poll_sync(self):
        self._poll_server_waiting = False
        self.monitor_syscalls(None, self._poll_ignore_callback, \
            self._poll_break_callback, self._poll_syscall_callback, \
            break_on_entry=True, timeout=self._data_timeout)
        del self._poll_server_waiting

    # this is needed so that poll_sync is executed by GLOBAL_ASYNC_EXECUTOR
    # so that the ptraced child is accessed by the same tracer
    @sync_to_async(executor=GLOBAL_ASYNC_EXECUTOR)
    def _async_poll_sync(self):
        return self._poll_sync()

    async def send(self, data: ByteString) -> int:
        # flush any data left in the receive buffer
        await self.receive()

        sent = 0
        while sent < len(data):
            sent += await self._send_sync(data[sent:])

        await self._async_poll_sync()
        debug(f"Sent data to server: {data}")
        return sent

    @sync_to_async(executor=GLOBAL_ASYNC_EXECUTOR)
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
                raise ChannelBrokenException("socket fd is negative, socket is closed") from ex

            ret = self._socket.recv(self.RECV_CHUNK_SIZE)
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
        try:
            if syscall.name in ('accept', 'accept4'):
                if syscall.arguments[0].value != listenfd \
                        or (syscall.result is not None and syscall.result < 0):
                    return
            # poll, ppoll
            elif syscall.name in ('poll', 'ppoll'):
                nfds = syscall.arguments[1].value
                pollfds = syscall.arguments[0].value
                fmt = '@ihh'
                size = struct.calcsize(fmt)
                for i in range(nfds):
                    fd, events, revents = struct.unpack(fmt, process.readBytes(pollfds + i * size, size))
                    if fd == listenfd and (events & select.POLLIN) != 0:
                        args = list(syscall.readArgumentValues(process.getregs()))
                        # convert call to blocking
                        args[2] = -1
                        syscall.writeArgumentValues(*args)
                        break
                else:
                    return
            # select
            elif syscall.name == 'select':
                nfds = syscall.arguments[0].value
                if nfds <= listenfd:
                    return
                readfds = syscall.arguments[1].value
                fmt = '@l'
                size = struct.calcsize(fmt)
                l_idx = listenfd // (size * 8)
                b_idx = listenfd % (size * 8)
                fd_set, = struct.unpack(fmt, process.readBytes(readfds + l_idx * size, size))
                if fd_set & (1 << b_idx) == 0:
                    return
                args = list(syscall.readArgumentValues(process.getregs()))
                # convert call to blocking
                args[4] = 0
                syscall.writeArgumentValues(*args)
            else:
                return

            self._setup_accepting = True
            self.cb_socket_accepting(process, syscall)
        finally:
            process.syscall()

    def _setup_ignore_callback_accept(self, syscall):
        return syscall.name not in ('accept', 'accept4', 'poll', 'ppoll', 'select')

    def _setup_break_callback_accept(self):
        return self._setup_accepting

    def _connect_syscall_callback(self, process, syscall, listenfd):
        if syscall.name in ('accept', 'accept4') \
                and syscall.arguments[0].value == listenfd \
                and syscall.result >= 0:
            self._sockfd.add(syscall.result)
            self._refcounter[syscall.result] = 1
            self.cb_socket_accepted(process, syscall)

    def _connect_ignore_callback(self, syscall):
        return syscall.name not in ('accept', 'accept4')

    def _connect_break_callback(self):
        return len(self._sockfd) > 0

    def _connect_monitor_target(self):
        return self._socket.connect(self._connect_address)

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
                    if fd in self._sockfd and (events & select.POLLIN) != 0:
                        args = list(syscall.readArgumentValues(process.getregs()))
                        # convert call to blocking
                        args[2] = -1
                        syscall.writeArgumentValues(*args)
                        break
                else:
                    return
            # select
            elif syscall.name == 'select':
                nfds = syscall.arguments[0].value
                if nfds <= max(self._sockfd):
                    return
                readfds = syscall.arguments[1].value
                fmt = '@l'
                size = struct.calcsize(fmt)
                for fd in self._sockfd:
                    l_idx = fd // (size * 8)
                    b_idx = fd % (size * 8)
                    fd_set, = struct.unpack(fmt, process.readBytes(readfds + l_idx * size, size))
                    if fd_set & (1 << b_idx) != 0:
                        break
                else:
                    return
                args = list(syscall.readArgumentValues(process.getregs()))
                # convert call to blocking
                args[4] = 0
                syscall.writeArgumentValues(*args)
            # read, recv, recvfrom, recvmsg
            elif syscall.name in ('read', 'recv', 'recvfrom', 'recvmsg') and syscall.arguments[0].value in self._sockfd:
                pass
            # For the next two cases, since break_on_entry is True, the effect
            # only takes place on the second occurrence, when the syscall exits;
            # so we check if the result is None (entry) or not (exit)
            elif syscall.name in ('shutdown', 'close') and syscall.arguments[0].value in self._sockfd:
                if syscall.result is None:
                    return
                elif syscall.name == 'shutdown':
                    raise ChannelBrokenException("Channel closed while waiting for server to read")
                elif syscall.name == 'close':
                    self._refcounter[syscall.arguments[0].value] -= 1
                    if sum(self._refcounter.values()) == 0:
                        raise ChannelBrokenException("Channel closed while waiting for server to read")
                    return
            elif syscall.name.startswith('dup') and syscall.arguments[0].value in self._sockfd:
                # FIXME might need to implement dup monitoring in all handlers
                # but for now this should work
                if syscall.result is None or syscall.result < 0:
                    return
                self._sockfd.add(syscall.result)
                self._refcounter[syscall.result] = 1
                debug(f"Accepted socket has been duplicated in fd={syscall.result}")
                return
            else:
                return
        finally:
            process.syscall()
        self._poll_server_waiting = True

    def _poll_ignore_callback(self, syscall):
        # TODO add support for epoll?
        return syscall.name not in ('read', 'recv', 'recvfrom', 'recvmsg', \
                                'poll', 'ppoll', 'select', 'close', 'shutdown',
                                'dup', 'dup2', 'dup3')

    def _poll_break_callback(self):
        return self._poll_server_waiting

    def _send_syscall_callback(self, process, syscall):
        if syscall.name in ('read', 'recv', 'recvfrom', 'recvmsg') \
                and syscall.arguments[0].value in self._sockfd:
            if syscall.result == 0:
                raise ChannelBrokenException("Server failed to read data off socket")
            self._send_server_received += syscall.result
        elif syscall.name == 'shutdown' and syscall.arguments[0].value in self._sockfd:
            raise ChannelBrokenException("Channel closed while waiting for server to read")
        elif syscall.name == 'close' and syscall.arguments[0].value in self._sockfd:
            self._refcounter[syscall.arguments[0].value] -= 1
            if sum(self._refcounter.values()) == 0:
                raise ChannelBrokenException("Channel closed while waiting for server to read")

    def _send_ignore_callback(self, syscall):
        return syscall.name not in ('read', 'recv', 'recvfrom', 'recvmsg', \
                                    'close', 'shutdown')

    def _send_break_callback(self):
        if not self._send_barrier.broken:
            self._send_barrier.wait()
            self._send_barrier.abort()
        debug(f"{self._send_client_sent=}; {self._send_server_received=}")
        # FIXME is there a case where client_sent == 0?
        if self._send_client_sent == 0 or self._send_server_received > self._send_client_sent:
            raise ChannelBrokenException("Client sent no bytes, or server received too many bytes!")
        return self._send_server_received == self._send_client_sent

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

class ListenerSocketState:
    SOCKET_UNBOUND = 1
    SOCKET_BOUND = 2
    SOCKET_LISTENING = 3

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
        elif self._state == self.SOCKET_BOUND and syscall.name == 'listen':
            if not entry:
                self._state = self.SOCKET_LISTENING
                debug(f"Server listening on port {self._sin_port}")
            return (self._sa_family, self._sin_addr, self._sin_port)
