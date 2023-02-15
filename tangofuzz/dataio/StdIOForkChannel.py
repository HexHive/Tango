from . import debug, info

from dataio import (ChannelBase,
                      PtraceForkChannel,
                      StdIOChannel, StdIOChannelFactory,
                      ChannelFactoryBase)
from subprocess import Popen
from dataclasses import dataclass
from functools import cached_property
from ptrace.syscall import SYSCALL_REGISTER
import os
import stat

@dataclass
class StdIOForkChannelFactory(StdIOChannelFactory):
    work_dir: str = None

    def create(self, pobj: Popen, *args, **kwargs) -> ChannelBase:
        self._pobj = pobj
        ch = self.forkchannel
        ch.connect()
        return ch

    @cached_property
    def forkchannel(self):
        return StdIOForkChannel(pobj=self._pobj,
                          work_dir=self.work_dir,
                          timescale=self.timescale)

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
            self.monitor_syscalls(None, \
                self._invoke_forkserver_ignore_callback, \
                self._invoke_forkserver_break_callback, \
                self._invoke_forkserver_syscall_callback, process=self._proc)
            self._awaited = True
            self._poll_sync()

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

    def _poll_sync(self):
        self._poll_server_waiting = False
        proc, _ = self.monitor_syscalls(self._poll_monitor, self._poll_ignore_callback, \
            self._poll_break_callback, self._poll_syscall_callback, \
            break_on_entry=True)
        del self._poll_server_waiting
        return proc

    ## Callbacks
    def _poll_monitor(self):
        if self._awaited and self._reconnect:
            self._file = open(self._fifo, 'wb', buffering=0)
            self._reconnect = False
