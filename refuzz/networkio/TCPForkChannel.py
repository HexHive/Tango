from . import debug, info

from typing import ByteString
from networkio import ChannelBase, PtraceForkChannel, TransportChannelFactory
from   common      import (ChannelBrokenException,
                          ChannelSetupException)
from subprocess import Popen
import errno
import struct
import socket
import select
import threading
from dataclasses import dataclass
from functools import partial, cached_property
from ptrace.error import PtraceError

@dataclass
class TCPForkChannelFactory(TransportChannelFactory):
    connect_timeout: float = None # seconds
    data_timeout: float = None # seconds

    def create(self, pobj: Popen) -> ChannelBase:
        self._pobj = pobj
        ch = self.forkchannel
        ch.connect((self.endpoint, self.port))
        return ch

    @cached_property
    def forkchannel(self):
        return TCPForkChannel(self._pobj,
                          self.endpoint, self.port,
                          timescale=self.timescale,
                          connect_timeout=self.connect_timeout,
                          data_timeout=self.data_timeout)


class TCPForkChannel(PtraceForkChannel):
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

    def __init__(self, pobj: Popen, endpoint: str, port: int, timescale: float,
                    connect_timeout: float, data_timeout: float):
        super().__init__(pobj, timescale)
        self._connect_timeout = connect_timeout * timescale if connect_timeout else None
        self._data_timeout = data_timeout * timescale if data_timeout else None
        self._socket = None 
        self._target = None
        self.setup((endpoint, port))

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
                        # matches our condition. Now, we should invoke the forkserver
                        self._invoke_forkserver(process)

        ## Wait for a socket that is listening on the same port
        # FIXME check for matching address too
        listeners = {}
        ignore_callback = lambda x: x.name not in ('socket', 'bind', 'listen')
        break_callback = lambda: any(x[2] == address[1] for x in listeners.values())

        self._monitor_syscalls(None, ignore_callback, break_callback, syscall_callback_listen, fds={}, listeners=listeners, timeout=self._connect_timeout)

    def connect(self, address: tuple):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ## Now that we've verified the server is listening, we can connect
        ## Listen for a call to accept() to get the connected socket fd
        sockfd = -1
        ignore_callback = lambda x: x.name not in ('accept', 'accept4')
        break_callback = lambda: sockfd > 0

        def syscall_callback_accept(process, syscall, listenfd):
            nonlocal sockfd
            if syscall.arguments[0].value != listenfd:
                return
            sockfd = syscall.result

        monitor_target = lambda: self._socket.connect(address)
        self._target, _ = self._monitor_syscalls(monitor_target, ignore_callback, break_callback, syscall_callback_accept, listenfd=self._listenfd, timeout=self._connect_timeout)
        debug("Socket is now connected!")
        self._sockfd = sockfd

        # wait for the next read, recv, select, or poll
        # or wait for the parent to fork, and trace child for these calls
        self._poll_sync()

    def _poll_sync(self):
        server_waiting = False
        # TODO add support for epoll?
        ignore_callback = lambda x: x.name not in ('read', 'recv', 'recvfrom', 'recvmsg', 'poll', 'ppoll', 'select')
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
            elif syscall.arguments[0].value != self._sockfd:
                return
            nonlocal server_waiting
            server_waiting = True

        self._monitor_syscalls(None, ignore_callback, break_callback, syscall_callback_read, break_on_entry=True, timeout=self._data_timeout)

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

        ignore_callback = lambda x: x.name not in ('read', 'recv', 'recvfrom', 'recvmsg')
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
            if syscall.arguments[0].value != self._sockfd:
                return
            nonlocal server_received
            if syscall.result == 0:
                raise ChannelBrokenException("Server failed to read data off socket")
            server_received += syscall.result

        monitor_target = lambda: send_monitor(data)
        _, ret = self._monitor_syscalls(monitor_target, ignore_callback, break_callback, syscall_callback_read, timeout=self._data_timeout)

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

    @property
    def current_target(self):
        return self._target

    def close(self, terminate=False):
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        super().close(terminate=terminate)