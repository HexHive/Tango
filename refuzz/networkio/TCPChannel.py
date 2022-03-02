from . import debug, info

from typing import ByteString
from networkio import ChannelBase, PtraceChannel, TransportChannelFactory
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
from ptrace.error import PtraceError

@dataclass
class TCPChannelFactory(TransportChannelFactory):
    connect_timeout: float = None # seconds
    data_timeout: float = None # seconds

    protocol: str = "tcp"

    def create(self, pobj: Popen) -> ChannelBase:
        ch = TCPChannel(pobj=pobj,
                          endpoint=self.endpoint, port=self.port,
                          timescale=self.timescale,
                          connect_timeout=self.connect_timeout,
                          data_timeout=self.data_timeout)
        ch.connect((self.endpoint, self.port))
        return ch

class TCPChannel(PtraceChannel):
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
                # maybe use the parser that comes with python-ptrace
                self._sa_family,  = struct.unpack('@H', process.readBytes(sockaddr, 2))
                assert (self._sa_family  == socket.AF_INET), "Only IPv4 is supported."
                self._sin_port, = struct.unpack('!H', process.readBytes(sockaddr + 2, 2))
                self._sin_addr = socket.inet_ntop(self._sa_family, process.readBytes(sockaddr + 4, 4))
                self._state = self.SOCKET_BOUND
                debug(f"Socket bound to port {self._sin_port}")
            elif self._state == self.SOCKET_BOUND and syscall.name == 'listen':
                self._state = self.SOCKET_LISTENING
                debug(f"Server listening on port {self._sin_port}")
                return (self._sa_family, self._sin_addr, self._sin_port)

    def __init__(self, endpoint: str, port: int,
                    connect_timeout: float, data_timeout: float, **kwargs):
        super().__init__(**kwargs)
        self._connect_timeout = connect_timeout * self._timescale if connect_timeout else None
        self._data_timeout = data_timeout * self._timescale if data_timeout else None
        self._socket = None
        self._accept_process = None
        self._refcounter = 0
        self.setup((endpoint, port))

    def cb_socket_listening(self, process):
        pass

    def cb_socket_accepted(self, process):
        self._refcounter = 1

    def process_new(self, *args, **kwargs):
        super().process_new(*args, **kwargs)
        # FIXME self._socket could be not None but the socket may not yet be
        # accepted (see: connect())
        if self._socket is not None:
            self._refcounter += 1

    def setup(self, address: tuple):
        def syscall_callback_listen(process, syscall, fds, listeners):
            if syscall.name == 'socket':
                domain = syscall.arguments[0].value
                typ = syscall.arguments[1].value
                if domain != socket.AF_INET or (typ & socket.SOCK_STREAM) == 0:
                    return
                fd = syscall.result
                fds[fd] = self.ListenerSocketState()
            elif syscall.name in ('bind', 'listen'):
                fd = syscall.arguments[0].value
                if fd not in fds:
                    return
                result = fds[fd].event(process, syscall)
                if result is not None:
                    listeners[fd] = result
                    candidates = [x[0] for x in listeners.items() if x[1][2] == address[1]]
                    if candidates:
                        self._listenfd = candidates[0]
                        # At this point, the target is paused just after the listen syscall that
                        # matches our condition.
                        self.cb_socket_listening(process)

        ## Wait for a socket that is listening on the same port
        # FIXME check for matching address too
        listeners = {}
        ignore_callback = lambda x: x.name not in ('socket', 'bind', 'listen')
        break_callback = lambda: any(x[2] == address[1] for x in listeners.values())

        self.monitor_syscalls(None, ignore_callback, break_callback, syscall_callback_listen, fds={}, listeners=listeners, timeout=self._connect_timeout)

    def connect(self, address: tuple):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ## Now that we've verified the server is listening, we can connect
        ## Listen for a call to accept() to get the connected socket fd
        sockfd = -1
        ignore_callback = lambda x: x.name not in ('accept', 'accept4')
        break_callback = lambda: sockfd > 0

        def syscall_callback_accept(process, syscall, listenfd):
            nonlocal sockfd
            if syscall.name in ('accept', 'accept4') and syscall.arguments[0].value == listenfd:
                sockfd = syscall.result

        monitor_target = lambda: self._socket.connect(address)
        self._accept_process, _ = self.monitor_syscalls(monitor_target, ignore_callback, break_callback, syscall_callback_accept, listenfd=self._listenfd, timeout=self._connect_timeout)
        self.cb_socket_accepted(self._accept_process)
        debug(f"Socket is now connected ({sockfd = })!")
        self._sockfd = sockfd

        # wait for the next read, recv, select, or poll
        # or wait for the parent to fork, and trace child for these calls
        self._poll_sync()

    def _poll_sync(self):
        server_waiting = False
        # TODO add support for epoll?
        ignore_callback = lambda x: x.name not in ('read', 'recv', 'recvfrom', 'recvmsg', 'poll', 'ppoll', 'select', 'close', 'shutdown')
        break_callback = lambda: server_waiting
        def syscall_callback_read(process, syscall):
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
            nonlocal server_waiting
            server_waiting = True

        self.monitor_syscalls(None, ignore_callback, break_callback, syscall_callback_read, break_on_entry=True, timeout=self._data_timeout)

    def send(self, data: ByteString) -> int:
        sent = 0
        while sent < len(data):
            sent += self._send_sync(data[sent:])

        self._poll_sync()
        debug(f"Sent data to server: {data}")
        return sent

    def _send_sync(self, data: ByteString) -> int:
        server_received = 0
        client_sent = 0
        ## Set up a barrier so that client_sent is ready when checking for break condition
        barrier = threading.Barrier(2)

        ignore_callback = lambda x: x.name not in ('read', 'recv', 'recvfrom', 'recvmsg', 'close', 'shutdown')
        def break_callback():
            if not barrier.broken:
                barrier.wait()
                barrier.abort()
            debug(f"{client_sent=}; {server_received=}")
            # FIXME is there a case where client_sent == 0?
            assert (client_sent > 0 and server_received <= client_sent), \
                "Client sent no bytes, or server received too many bytes!"
            return server_received == client_sent

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
                # but it's benign
                pass
            return ret

        def syscall_callback_read(process, syscall):
            if syscall.name in ('read', 'recv', 'recvfrom', 'recvmsg') and syscall.arguments[0].value == self._sockfd:
                nonlocal server_received
                if syscall.result == 0:
                    raise ChannelBrokenException("Server failed to read data off socket")
                server_received += syscall.result
            elif syscall.name == 'shutdown' and syscall.arguments[0].value == self._sockfd:
                raise ChannelBrokenException("Channel closed while waiting for server to read")
            elif syscall.name == 'close' and syscall.arguments[0].value == self._sockfd:
                self._refcounter -= 1
                if self._refcounter == 0:
                    raise ChannelBrokenException("Channel closed while waiting for server to read")

        monitor_target = lambda: send_monitor(data)
        _, ret = self.monitor_syscalls(monitor_target, ignore_callback, break_callback, syscall_callback_read, timeout=self._data_timeout)

        return ret

    def receive(self) -> ByteString:
        chunks = []
        while True:
            try:
                poll, _, _ = select.select([self._socket], [], [], 0)
                if self._socket not in poll:
                    data = b''.join(chunks)
                    debug(f"Received data from server: {data}")
                    return data
            except ValueError:
                raise ChannelBrokenException("socket fd is negative, socket is closed")

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
