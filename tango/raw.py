from . import debug

from tango.core import (AbstractChannel, FormatDescriptor, AbstractInstruction,
    TransmitInstruction, SerializedInputMeta, CountProfiler)
from tango.ptrace.syscall import PtraceSyscall, SYSCALL_REGISTER
from tango.ptrace.debugger import PtraceProcess
from tango.unix import (PtraceChannel, PtraceForkChannel, PtraceChannelFactory,
    FileDescriptorChannel, FileDescriptorChannelFactory)
from tango.exceptions import ChannelBrokenException
from tango.common import sync_to_async, GLOBAL_ASYNC_EXECUTOR

from typing import ByteString, Iterable, Mapping, Any
from subprocess import Popen
from dataclasses import dataclass
from functools import cached_property
import errno
import struct
import select
import threading
import os
import stat

__all__ = [
    'StdIOChannelFactory', 'StdIOChannel', 'StdIOForkChannelFactory',
    'StdIOForkChannel', 'RawInput'
]

@dataclass(kw_only=True, frozen=True)
class StdIOChannelFactory(FileDescriptorChannelFactory):
    fmt: FormatDescriptor = FormatDescriptor('raw')

    def create(self, pobj: Popen, netns: str) -> AbstractChannel:
        ch = StdIOChannel(pobj=pobj, **self.fields)
        ch.connect()
        return ch

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['channel'].get('type') == 'stdio'

class StdIOChannel(FileDescriptorChannel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stdinfd = -1
        self._file = None

    def cb_stdin_polled(self, process, syscall):
        pass

    def sync_callback(self, process: PtraceProcess, syscall: PtraceSyscall):
        super().sync_callback(process, syscall)
        self.cb_stdin_polled(process, syscall)

    def _reset_refs(self):
        self._refcounter.clear()
        self._stdinfd = 0
        self._refcounter[0] = 1

    def connect(self):
        self._reset_refs()
        self._file = self._pobj.stdin
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
            partial = await self._send_sync(data[sent:])
            sent += partial
            CountProfiler('bytes_sent')(partial)

        await self.a_sync()
        assert self.synced
        debug(f"Sent data to server: {data}")

        # we do this since TransmitInteraction.perform already counts sent bytes
        CountProfiler('bytes_sent')(-sent)
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
            self._send_syscall_callback)
        return ret

    async def receive(self) -> ByteString:
        # FIXME for now, stdout/stderr are not piped
        return b''

    def close(self, **kwargs):
        if self._file is not None and not self._file.closed:
            self._file.flush()
        super().close(**kwargs)

    ### Callbacks ###
    def _send_syscall_callback(self, process, syscall):
        if syscall.name == 'read' and \
                syscall.arguments[0].value in self._refcounter:
            if syscall.result == 0:
                raise ChannelBrokenException(
                    "Server failed to read data off stdin")
            self._send_server_received += syscall.result

    def _send_ignore_callback(self, syscall):
        return syscall.name != 'read'

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
        if self._send_client_sent == 0 or \
                self._send_server_received > self._send_client_sent:
            raise ChannelBrokenException(
                "Client sent no bytes, or server received too many bytes!")
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

@dataclass(kw_only=True, frozen=True)
class StdIOForkChannelFactory(StdIOChannelFactory,
        capture_paths=['fuzzer.work_dir']):
    work_dir: str = None

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['driver'].get('forkserver')

    def create(self, pobj: Popen, netns: str) -> AbstractChannel:
        object.__setattr__(self, '_pobj', pobj)
        ch = self.forkchannel
        ch.connect()
        return ch

    @cached_property
    def forkchannel(self):
        return StdIOForkChannel(pobj=self._pobj, **self.fields)

class StdIOForkChannel(StdIOChannel, PtraceForkChannel):
    def __init__(self, work_dir: str, **kwargs):
        super().__init__(**kwargs)
        self._work_dir = work_dir
        self._injected = False
        self._awaited = False
        self._reconnect = True
        self._fifo = os.path.join(self._work_dir, 'input.pipe')
        self._setup_fifo()

    def connect(self):
        self._reconnect = True
        super().connect()

        if not self._awaited:
            # FIXME it is preferred that this be part of PtraceForkChannel, but
            # it can only be done outside the call frame of an existing
            # monitor_syscalls call.
            self.monitor_syscalls(None,
                self._invoke_forkserver_ignore_callback,
                self._invoke_forkserver_break_callback,
                self._invoke_forkserver_syscall_callback, process=self._proc)
            self._awaited = True
            self.sync()
            assert self.synced

    def close(self, **kwargs):
        if self._file is not None and not self._file.closed:
            self._file.close()
            self._file = None
        super().close(**kwargs)

    def cb_stdin_polled(self, process, syscall):
        if not self._injected:
            self._invoke_forkserver_custom(process, syscall)
            self._injected = True

    def _setup_fifo(self):
        if os.path.exists(self._fifo):
            if stat.S_ISFIFO(os.stat(self._fifo).st_mode):
                return
            else:
                os.unlink(self._fifo)
        os.mkfifo(self._fifo, 0o600)

    def _invoke_forkserver_custom(self, process, syscall):
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

    def sync(self):
        self.synced = False
        ignore_cb = lambda s: True
        break_cb = lambda: self.synced
        syscall_cb = lambda p, s: False # will never be called
        self.monitor_syscalls(self._sync_monitor, ignore_cb, break_cb, syscall_cb,
            break_on_entry=False, break_on_exit=False,
            timeout=self._data_timeout)

    ## Callbacks
    def _sync_monitor(self):
        if self._awaited and self._reconnect:
            self._file = open(self._fifo, 'wb', buffering=0)
            self._reconnect = False

class RawInput(metaclass=SerializedInputMeta, typ='raw'):
    CHUNKSIZE = 4

    def loadi(self) -> Iterable[AbstractInstruction]:
        data = self._file.read()
        unpack_len = len(data) - (len(data) % self.CHUNKSIZE)
        for s, in struct.iter_unpack(f'{self.CHUNKSIZE}s', data[:unpack_len]):
            instruction = TransmitInstruction(data=s)
            yield instruction
        if unpack_len < len(data):
            instruction = TransmitInstruction(data=data[unpack_len:])
            yield instruction

    def dumpi(self, itr: Iterable[AbstractInstruction], /):
        for instruction in itr:
            if isinstance(instruction, TransmitInstruction):
                self._file.write(instruction._data)

    def __repr__(self):
        return f"RawInput({self._file})"
