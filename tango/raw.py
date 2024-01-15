from . import debug

from tango.core import (AbstractChannel, FormatDescriptor, AbstractInstruction,
    TransmitInstruction, SerializedInputMeta, CountProfiler)
from tango.ptrace.syscall import PtraceSyscall, SYSCALL_REGISTER
from tango.ptrace.debugger import PtraceProcess
from tango.unix import (PtraceChannel, PtraceForkChannel, PtraceChannelFactory,
    FileDescriptorChannel, FileDescriptorChannelFactory, ChunkSizeDescriptor)
from tango.exceptions import ChannelBrokenException, NotSyncedException
from tango.common import sync_to_async

from typing import ByteString, Iterable, Mapping, Any, Optional
from subprocess import Popen
from dataclasses import dataclass
from pathlib import Path
import errno
import struct
import select
import os
import stat
import fcntl

__all__ = [
    'StdIOChannelFactory', 'StdIOChannel', 'StdIOForkChannelFactory',
    'StdIOForkChannel', 'RawInput'
]

@dataclass(kw_only=True, frozen=True)
class RawFormatDescriptor(FormatDescriptor):
    typ: str = 'raw'
    chunk_size: Optional[int] = None

    def __get__(self, obj, owner):
        if obj is None:
            return None
        return getattr(obj, '_fmt')

    def __set__(self, obj, value):
        if not value:
            return
        fmt = type(self)(chunk_size=value[0])
        object.__setattr__(obj, '_fmt', fmt)

@dataclass(kw_only=True, frozen=True)
class StdIOChannelFactory(FileDescriptorChannelFactory,
        capture_paths=('channel.chunk_size',)):
    fmt: FormatDescriptor = RawFormatDescriptor()
    chunk_size: ChunkSizeDescriptor = ChunkSizeDescriptor()

    def __post_init__(self):
        # implicit casting through the descriptor
        object.__setattr__(self, 'fmt', (self.chunk_size,))

    async def create(self, pobj: Popen, netns: str, **kwargs) -> AbstractChannel:
        ch = StdIOChannel(pobj=pobj, **self.fields, **kwargs)
        await ch.setup()
        await ch.connect()
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

    @property
    def file(self):
        return self._file

    async def connect(self):
        self._reset_refs()
        self._file = self._pobj.stdin
        # wait for the next read, recv, select, or poll
        # or wait for the parent to fork, and trace child for these calls
        await self.sync()
        if not self.synced:
            raise NotSyncedException("Failed to sync after connection")

    async def read_bytes(self) -> ByteString:
        # FIXME for now, stdout/stderr are not piped
        return b''

    async def write_bytes(self, data: ByteString) -> int:
        return self._file.write(data)

    def cb_stdin_polled(self, process, syscall):
        pass

    def sync_callback(self, process: PtraceProcess, syscall: PtraceSyscall):
        super().sync_callback(process, syscall)
        self.cb_stdin_polled(process, syscall)

    def _reset_refs(self):
        self._refcounter.clear()
        self._stdinfd = 0
        self._refcounter[0] = 1
        debug(f"Clear the fd refcounter")

    async def shutdown(self):
        if self._file is not None and not self._file.closed:
            # FIXME should close self._file here?
            self._file.flush()
            self._file.close()
            self._file = None

    async def close(self):
        await self.shutdown()
        await super().close()

@dataclass(kw_only=True, frozen=True)
class StdIOForkChannelFactory(StdIOChannelFactory,
        capture_paths=['fuzzer.work_dir']):
    work_dir: str
    _cached = None

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['driver'].get('forkserver')

    async def create(self, pobj: Popen, netns: str, **kwargs) -> AbstractChannel:
        ch = await self._create_once(pobj=pobj, **kwargs)
        await ch.connect()
        return ch

    async def _create_once(self, **kwargs):
        if self._cached is not None:
            return self._cached
        ch = StdIOForkChannel(**kwargs, **self.fields)
        object.__setattr__(self, '_cached', ch)
        await ch.setup()
        return ch

class StdIOForkChannel(StdIOChannel, PtraceForkChannel):
    def __init__(self, work_dir: str, **kwargs):
        super().__init__(blocking=False, **kwargs)
        self._work_dir = work_dir
        self._injected = False
        self._awaited = False
        self._reconnect = True

        self._fifo = Path(self._work_dir) / 'shared/input.pipe'
        self._setup_fifo()

    async def connect(self):
        self._reconnect = True
        await super().connect()

        if not self._awaited:
            # FIXME it is preferred that this be part of PtraceForkChannel, but
            # it can only be done outside the call frame of an existing
            # monitor_syscalls call.
            await self.monitor_syscalls(None,
                self._invoke_forkserver_ignore_callback,
                self._invoke_forkserver_break_callback,
                self._invoke_forkserver_syscall_callback, process=self._proc)
            self._awaited = True
            await self.sync()
            if not self.synced:
                raise NotSyncedException("Failed to sync after connection")

    def cb_stdin_polled(self, process, syscall):
        if not self._injected:
            self._invoke_forkserver_custom(process, syscall)
            self._injected = True

    def _setup_fifo(self):
        if self._fifo.exists():
            if stat.S_ISFIFO(self._fifo.stat().st_mode):
                return
            else:
                self._fifo.unlink()
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

    async def sync(self):
        async def break_cb():
            return self.synced
        async def syscall_cb(syscall):
            assert False # will never be called
        self.synced = False
        ignore_cb = lambda s: True
        await self.monitor_syscalls(
            self._sync_monitor_target, ignore_cb, break_cb, syscall_cb,
            break_on_entry=False, break_on_exit=False,
            timeout=self._data_timeout)

    ## Callbacks
    @sync_to_async()
    def _sync_monitor_target(self):
        if self._awaited and self._reconnect:
            self._file = open(self._fifo, 'wb', buffering=0)
            # at this point, the pipe has been opened; we'll make it
            # non-blocking
            fd = self._file.fileno()
            flag = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flag | os.O_NONBLOCK)
            self._reconnect = False

class RawInput(metaclass=SerializedInputMeta, typ='raw'):
    def loadi(self) -> Iterable[AbstractInstruction]:
        data = self._file.read()
        if (chunk := self._fmt.chunk_size):
            unpack_len = len(data) - (len(data) % chunk)
            for s, in struct.iter_unpack(f'{chunk}s', data[:unpack_len]):
                instruction = TransmitInstruction(data=s)
                yield instruction
        else:
            unpack_len = chunk = 0

        if unpack_len < len(data):
            instruction = TransmitInstruction(data=data[unpack_len:])
            yield instruction

    def dumpi(self, itr: Iterable[AbstractInstruction], /):
        for instruction in itr:
            if isinstance(instruction, TransmitInstruction):
                self._file.write(instruction._data)

    def __repr__(self):
        return f"RawInput({self._file})"
