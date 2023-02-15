from . import debug, info

from typing import ByteString
from dataio import ChannelBase, ChannelFactoryBase, PtraceChannel
from   common      import (ChannelBrokenException,
                          ChannelSetupException)
from subprocess import Popen
import errno
import struct
import select
import threading
from dataclasses import dataclass
from functools import partial
from ptrace import PtraceError
from common import sync_to_async, GLOBAL_ASYNC_EXECUTOR
from input import FormatDescriptor

@dataclass(kw_only=True)
class StdIOChannelFactory(ChannelFactoryBase):
    fmt: FormatDescriptor = FormatDescriptor('raw')

    def create(self, pobj: Popen, *args, **kwargs) -> ChannelBase:
        ch = StdIOChannel(pobj=pobj,
                          timescale=self.timescale)
        ch.connect()
        return ch

class StdIOChannel(PtraceChannel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fd = set()
        self._file = None
        self._refcounter = dict()

    def process_new(self, *args, **kwargs):
        super().process_new(*args, **kwargs)
        for fd in self._refcounter:
            self._refcounter[fd] += 1

    def cb_stdin_polled(self, process, syscall):
        pass

    def _reset_refs(self):
        self._fd.clear()
        self._fd.add(0)
        self._refcounter[0] = 1

    def connect(self):
        self._reset_refs()
        self._file = self._pobj.stdin
        # wait for the next read, recv, select, or poll
        # or wait for the parent to fork, and trace child for these calls
        self._poll_sync()

    def _poll_sync(self):
        self._poll_server_waiting = False
        proc, _ = self.monitor_syscalls(None, self._poll_ignore_callback, \
            self._poll_break_callback, self._poll_syscall_callback, \
            break_on_entry=True)
        del self._poll_server_waiting
        return proc

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
        self._send_barrier_passed = False

        self._send_data = data
        _, ret = self.monitor_syscalls(self._send_send_monitor, \
            self._send_ignore_callback, self._send_break_callback, \
            self._send_syscall_callback)

        del self._send_server_received
        del self._send_client_sent
        del self._send_barrier
        del self._send_barrier_passed
        del self._send_data

        return ret

    async def receive(self) -> ByteString:
        # FIXME for now, stdout/stderr are not piped
        return b''

    def close(self, **kwargs):
        if self._file is not None and not self._file.closed:
            self._file.flush()
        super().close(**kwargs)

    ### Callbacks ###
    def _poll_syscall_callback(self, process, syscall):
        try:
            self._poll_syscall_callback_internal(process, syscall)
            if self._poll_server_waiting:
                self.cb_stdin_polled(process, syscall)
        finally:
            process.syscall()

    def _poll_syscall_callback_internal(self, process, syscall):
        # poll, ppoll
        if syscall.name in {'poll', 'ppoll'}:
            nfds = syscall.arguments[1].value
            pollfds = syscall.arguments[0].value
            fmt = '@ihh'
            size = struct.calcsize(fmt)
            for i in range(nfds):
                fd, events, revents = struct.unpack(fmt, process.readBytes(pollfds + i * size, size))
                if fd in self._fd and (events & select.POLLIN) != 0:
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
            if nfds <= max(self._fd):
                return
            readfds = syscall.arguments[1].value
            fmt = '@l'
            size = struct.calcsize(fmt)
            for fd in self._fd:
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
        elif syscall.name == 'read' and syscall.arguments[0].value in self._fd:
            pass
        # For the next two cases, since break_on_entry is True, the effect
        # only takes place on the second occurrence, when the syscall exits;
        # so we check if the result is None (entry) or not (exit)
        elif syscall.name == 'close' and syscall.arguments[0].value in self._fd:
            if syscall.result is None:
                return
            else:
                self._refcounter[syscall.arguments[0].value] -= 1
                if sum(self._refcounter.values()) == 0:
                    raise ChannelBrokenException("Channel closed while waiting for server to read")
                return
        elif syscall.name.startswith('dup') and syscall.arguments[0].value in self._fd:
            # FIXME might need to implement dup monitoring in all handlers
            # but for now this should work
            if syscall.result is None or syscall.result < 0:
                return
            self._fd.add(syscall.result)
            self._refcounter[syscall.result] = 1
            debug(f"stdin has been duplicated in fd={syscall.result}")
            return
        else:
            return
        self._poll_server_waiting = True

    def _poll_ignore_callback(self, syscall):
        # TODO add support for epoll?
        return syscall.name not in {'read', 'poll', 'ppoll', 'select', 'close', \
                                'dup', 'dup2', 'dup3'}

    def _poll_break_callback(self):
        return self._poll_server_waiting

    def _send_syscall_callback(self, process, syscall):
        if syscall.name == 'read' and syscall.arguments[0].value in self._fd:
            if syscall.result == 0:
                raise ChannelBrokenException("Server failed to read data off stdin")
            self._send_server_received += syscall.result
        elif syscall.name == 'close' and syscall.arguments[0].value in self._fd:
            self._refcounter[syscall.arguments[0].value] -= 1
            if sum(self._refcounter.values()) == 0:
                raise ChannelBrokenException("Channel closed while waiting for server to read")

    def _send_ignore_callback(self, syscall):
        return syscall.name not in {'read', 'close'}

    def _send_break_callback(self):
        if not self._send_barrier_passed:
            try:
                self._send_barrier.wait()
            except threading.BrokenBarrierError:
                raise ChannelBrokenException("Barrier broke while waiting")
            else:
                self._send_barrier_passed = True
        debug(f"{self._send_client_sent=}; {self._send_server_received=}")
        # FIXME is there a case where client_sent == 0?
        if self._send_client_sent == 0 or self._send_server_received > self._send_client_sent:
            raise ChannelBrokenException("Client sent no bytes, or server received too many bytes!")
        return self._send_server_received == self._send_client_sent

    def _send_send_monitor(self):
        try:
            ret = self._file.write(self._send_data)
            if ret == 0:
                raise ChannelBrokenException("Failed to send any data")
        except Exception:
            self._send_barrier.abort()
            raise
        self._send_client_sent = ret
        self._send_barrier.wait()
        return ret
