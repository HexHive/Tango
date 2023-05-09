from __future__ import annotations

from . import debug, info, warning, critical, error

from tango.core import (BaseDriver, AbstractState, Transition,
    LoadableTarget, ValueProfiler, CountProfiler, AbstractChannel,
    AbstractChannelFactory)
from tango.ptrace.binding import (HAS_SECCOMP_FILTER, HAS_NAMESPACES,
    ptrace_traceme, prctl)
from tango.ptrace.cpu_info import CPU_WORD_SIZE
from tango.ptrace.binding.cpu import CPU_INSTR_POINTER, CPU_STACK_POINTER
from tango.ptrace.debugger import   (PtraceDebugger, PtraceProcess,
    ProcessEvent, ProcessExit, ProcessSignal, NewProcessEvent, ProcessExecution,
    ForkChildKilledEvent)
from tango.ptrace import PtraceError
from tango.ptrace.func_call import FunctionCallOptions
from tango.ptrace.syscall import PtraceSyscall, SOCKET_SYSCALL_NAMES
from tango.ptrace.tools import signal_to_exitcode
from tango.ptrace.linux_proc import readProcessStat
from tango.common import ComponentType
from tango.exceptions import (ChannelTimeoutException, StabilityException,
    ProcessCrashedException, ProcessTerminatedException, ChannelBrokenException)

if HAS_SECCOMP_FILTER:
    from tango.ptrace.binding import (BPF_LD, BPF_W, BPF_ABS, BPF_JMP, BPF_JEQ,
        BPF_K, BPF_RET, BPF_STMT, BPF_JUMP, BPF_PROG, BPF_FILTER,
        SECCOMP_RET_ALLOW, SECCOMP_RET_TRACE, SECCOMP_RET_DATA,
        SECCOMP_FILTER_FLAG_TSYNC, SECCOMP_SET_MODE_FILTER,
        PR_SET_NO_NEW_PRIVS, PR_SET_PDEATHSIG)
    from tango.ptrace.binding.linux_struct import seccomp_data
    from tango.ptrace.syscall import SYSCALL_NUMBERS as NR

if HAS_NAMESPACES:
    from tango.ptrace.binding import (mount, umount, unshare, pivot_root,
        has_caps, drop_caps,
        MS_BIND, MS_REMOUNT, MS_RDONLY, MS_NOATIME, MS_NODEV, MS_NOSUID, MS_REC,
        MS_PRIVATE, CLONE_NEWNS, CAP_CHOWN, CAP_DAC_OVERRIDE, CAP_FOWNER,
        CAP_SETPCAP, CAP_NET_ADMIN, CAP_SYS_ADMIN)

from pathlib import Path
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import (
    IO, AsyncGenerator, Callable, Iterable, Optional, Any, ByteString,
    Awaitable)
from pyroute2 import netns, IPRoute
from subprocess import Popen, TimeoutExpired, DEVNULL, PIPE
from elftools.elf.elffile import ELFFile
from uuid import uuid4
from string import ascii_letters, digits
from functools import partial
import os
import sys
import ctypes
import signal
import traceback
import struct
import select
import posix_ipc
import mmap
import asyncio

__all__ = [
    'ProcessDriver', 'ProcessForkDriver',
    'PtraceChannel', 'PtraceForkChannel', 'PtraceChannelFactory',
    'FileDescriptorChannel', 'FileDescriptorChannelFactory',
    'SharedMemoryObject', 'resolve_symbol', 'EventOptions'
]

SOCKET_SYSCALL_NAMES = SOCKET_SYSCALL_NAMES.union(('read', 'write'))

@dataclass
class EventOptions:
    ignore_callback: Callable[[PtraceSyscall], bool] = lambda s: True
    syscall_callback: Callable[[PtraceProcess, PtraceSyscall], Awaitable] = None
    break_on_entry: bool = False
    break_on_exit: bool = False

DefaultOptions = EventOptions()

class PtraceChannel(AbstractChannel):
    def __init__(self, pobj: Popen, *, use_seccomp: bool,
            on_syscall_exception: Optional[Callable[
                ProcessEvent, PtraceSyscall, Exception]]=None,
            loop=None, **kwargs):
        super().__init__(**kwargs)
        self._pobj = pobj
        self._use_seccomp = use_seccomp

        # DEBUG options
        self._process_all = False
        self._verbose = False

        debug("Setting up new ptrace-enabled channel")

        self._syscall_options = FunctionCallOptions(
            write_types=False,
            write_argname=False,
            string_max_length=300,
            replace_socketcall=True,
            write_address=False,
            max_array_count=20,
        )
        self._syscall_options.instr_pointer = False

        self._debugger = PtraceDebugger()
        self._debugger.traceFork()
        self._debugger.traceExec()
        self._debugger.traceClone()
        if use_seccomp:
            self._debugger.traceSeccomp()

        self.observed = {}
        self.on_syscall_exception = on_syscall_exception

    async def setup(self):
        # we setup a catch-all subscription
        self._dbgsub = self._debugger.subscribe()
        self._proc = await self._debugger.addProcess(self._pobj.pid,
            is_attached=True, subscription=self._dbgsub)
        self.prepare_process(self._proc, resume=True)

    @property
    def root(self):
        return self._proc

    @root.setter
    def root(self, value):
        self._proc = value

    def prepare_process(self, process, *,
            ignore_callback=None, mask=None, resume=True):
        if not process in self._debugger:
            self._debugger.traceProcess(process)

        if self._process_all:
            ignore_callback = lambda x: False
        if mask:
            ignore_callback = mask
        elif opts := self.observed.get(process.root):
            ignore_callback = opts[1].ignore_callback

        if ignore_callback:
            process.syscall_state.ignore_callback = ignore_callback

        if resume:
            self.resume_process(process)

    async def process_syscall(self, process, syscall, syscall_callback, is_entry, **kwargs) -> bool:
        # ensure that the syscall has finished successfully before callback
        if is_entry or syscall.result >= 0:
            debug("syscall requested: [%i] %s", process.pid, syscall)
            processed = await syscall_callback(process, syscall, **kwargs)
        else:
            processed = False
        return processed

    async def process_exit(self, event):
        debug("Process with pid=%i exited, deleting from debugger", event.process.pid)
        debug("Reason: %s", event)
        event.process.deleteFromDebugger()
        try:
            if event.exitcode == 0:
                raise ProcessTerminatedException(
                    f"Process with {event.process.pid=} exited normally",
                    exitcode=0)
            elif event.signum is not None and \
                    event.signum not in (signal.SIGTERM, signal.SIGKILL):
                raise ProcessCrashedException(
                    f"Process with {event.process.pid=} crashed"
                    f" with {event.signum=}", signum=event.signum)
            else:
                raise ProcessTerminatedException(
                    f"Process with {event.process.pid=} terminated abnormally"
                    f" with {event.exitcode=}", exitcode=event.exitcode)
        finally:
            if event.process.root in self.observed:
                await self.pop_observe(event.process.root, wait_exit=False)

    async def process_signal(self, event):
        if event.signum in (signal.SIGINT, signal.SIGWINCH):
            debug("Process with pid=%i received SIGINT or SIGWINCH signum=%i",
                event.process.pid, event.signum)
            # Ctrl-C or resizing the window should not be passed to child
            self.resume_process(event.process)
            return
        elif event.signum == signal.SIGSTOP:
            critical("Process pid=%i received rogue SIGSTOP, resuming for now",
                event.process.pid)
            self.resume_process(event.process)
            return
        elif event.signum == signal.SIGSEGV:
            raise ProcessCrashedException(f"Process with {event.process.pid=} terminated abnormally with {event.signum=}", exitcode=event.signum)
        debug("Target process with pid=%i received signal with signum=%i",
            event.process.pid, event.signum)
        self.resume_process(event.process, event.signum)

    async def process_new(self, event, ignore_callback):
        # monitor child for syscalls as well. may be needed for multi-thread or multi-process targets
        debug("Target process with pid=%i forked, adding child process with pid=%i to debugger",
            event.process.parent.pid, event.process.pid)
        if event.process.is_attached:
            # sometimes, child process might have been killed at creation,
            # so the debugger detaches it; we check for that here
            self.prepare_process(
                event.process, ignore_callback=ignore_callback, resume=True)
        self.resume_process(event.process.parent)

    async def process_exec(self, event):
        debug("Target process with pid=%i called exec; removing from debugger",
            event.process.pid)
        event.process.detach()

    async def process_auxiliary_event(self, event, ignore_callback):
        try:
            raise event
        except ProcessExit as event:
            await self.process_exit(event)
        except ProcessSignal as event:
            await self.process_signal(event)
        except NewProcessEvent as event:
            await self.process_new(event, ignore_callback)
        except ProcessExecution as event:
            await self.process_exec(event)

    async def process_event(self, event: ProcessEvent, opts: EventOptions,
            **kwargs):
        if event is None:
            return

        is_syscall = event.is_syscall_stop()
        if not is_syscall:
            await self.process_auxiliary_event(event, opts.ignore_callback)
        else:
            # Process syscall enter or exit
            state = event.process.syscall_state

            ### DEBUG ###
            if self._verbose:
                debug("Target process with pid=%i encountered a syscall-stop",
                    event.process.pid)
                sc = PtraceSyscall(event.process, self._syscall_options,
                                   event.process.getregs())
                debug("syscall traced: [%i] sc.name=%s state.name=%s state.next_event=%s",
                    event.process.pid, sc.name, state.name, state.next_event)
            #############

            syscall = await state.event(self._syscall_options)
            is_entry = state.next_event == 'exit'
            is_exit = not is_entry

            match_condition = (opts.break_on_entry and is_entry) or \
                (opts.break_on_exit and is_exit)
            ignore_condition = syscall is not None and \
                not (self._process_all and opts.ignore_callback(syscall))
            should_process = ignore_condition and match_condition

            processed = False
            try:
                if should_process:
                    processed = await self.process_syscall(event.process, syscall,
                        opts.syscall_callback, is_entry, **kwargs)
            except Exception as ex:
                if self.on_syscall_exception:
                    self.on_syscall_exception(event, syscall, ex)
                raise
            finally:
                if not processed:
                    if is_entry:
                        # we need to catch the next syscall exit, so we must
                        # call process.syscall();
                        # with seccomp enabled, process.cont() is called
                        # instead, and we would not catch the exit
                        event.process.syscall()
                    else:
                        # resume the suspended process until it encounters the
                        # next syscall
                        self.resume_process(event.process)

            return syscall

    def resume_process(self, process, signum=0):
        if self._use_seccomp:
            process.cont(signum)
        else:
            process.syscall(signum)

    async def push_observe(self,
            callback: Callable[[ProcessEvent, Exception], Any],
            opts: EventOptions=DefaultOptions) -> Optional[PtraceProcess]:
        if not (root := self.root):
            return
        self.observed[root] = callback, opts
        for proc in self._debugger:
            if proc.root == root:
                # we ignore all syscalls for existing observed processes
                self.prepare_process(proc,
                    mask=opts.ignore_callback, resume=False)
        self.root = None
        return root

    async def pop_observe(self, root: PtraceProcess, kill: bool=True,
            wait_exit: bool=True, timeout: Optional[int]=None) -> Optional[
            tuple[Callable[[ProcessEvent, Exception], Any], EventOptions]]:
        if (rv := self.observed.get(root, None)):
            try:
                if kill:
                    if wait_exit:
                        for proc in self._debugger:
                            if proc.root is root:
                                proc.forkSubscription()
                    coro = root.terminateTree(
                        wait_exit=wait_exit, signum=signal.SIGKILL)
                    if timeout:
                        coro = asyncio.wait_for(coro, timeout)
                    await coro
            except asyncio.TimeoutError:
                pass
            finally:
                # We only pop it from observed after it is terminated;
                # otherwise, during the `await terminate`, another task might be
                # scheduled, and it may be running `monitor_process`, in which
                # `root` would not be observed, but it may still be in the
                # debugger. It is then considered a normal process, and it may
                # try to `ptrace` it, but it would failed because the process
                # is SIGKILLed
                debug("%s is no longer under observation", root)
                self.observed.pop(root, None)
        return rv

    async def _wait_for_syscall(self, process: PtraceProcess=None, **kwargs):
        kwargs['subscription'] = self._dbgsub
        return await self._debugger.waitProcessEvent(process, **kwargs)

    async def _monitor_syscalls_internal_loop(self,
            ignore_callback: Callable[[PtraceSyscall], bool],
            break_callback: Callable[..., bool],
            syscall_callback: Callable[[PtraceProcess, PtraceSyscall], None],
            break_on_entry: bool = False, break_on_exit: bool = True,
            process: PtraceProcess = None,
            **kwargs):
        monitor_opts = EventOptions(
            ignore_callback, syscall_callback,
            break_on_entry, break_on_exit)

        last_process = None
        monitoring = True
        while monitoring:
            try:
                if not self._debugger:
                    raise ProcessTerminatedException(
                        "Process was terminated while waiting for syscalls",
                        exitcode=None)

                debug("Waiting for syscall...")
                event = await self._wait_for_syscall(process)
                if not event:
                    continue

                if (observe_opts := self.observed.get(event.process.root)):
                    debug("Masking event for %s with root=%s",
                        event.process, event.process.root)
                    observe_cb, opts = observe_opts
                elif event.process.root not in self._debugger and \
                        event.process.root != self.root:
                    debug("Received rogue event (%s) for %s", event, event.process)
                    continue
                else:
                    opts = monitor_opts

                try:
                    rv = await self.process_event(event, opts, **kwargs)
                    if rv and not observe_opts and event.is_syscall_stop():
                        last_process = event.process
                except Exception as ex:
                    if observe_opts:
                        observe_cb(event, ex)
                    else:
                        raise

                if await break_callback():
                    debug("Syscall monitoring finished, breaking out of debug loop")
                    monitoring = False
            except asyncio.CancelledError:
                break
        else:
            return last_process

    async def monitor_syscalls(self,
            monitor_target: Callable,
            ignore_callback: Callable[[PtraceSyscall], bool],
            break_callback: Callable[..., bool],
            syscall_callback: Callable[[PtraceProcess, PtraceSyscall], None], /,
            timeout: float = None,
            process: PtraceProcess = None,
            **kwargs):
        # DEBUG
        if self._process_all:
            ignore_callback = lambda x: False
        procs = (process,) if process else self._debugger
        for proc in procs:
            # update the ignore_callback of processes in the debugger
            self.prepare_process(
                proc, ignore_callback=ignore_callback, resume=False)

        ## Prepare coroutine
        tasks = [self._monitor_syscalls_internal_loop(
            ignore_callback, break_callback, syscall_callback,
            process=process, **kwargs)]

        ## Execute monitor target
        if monitor_target:
            tasks.append(monitor_target())

        gather = asyncio.gather(*tasks)

        ## Wrap coroutine in timer if necessary
        if timeout is not None:
            gather = asyncio.wait_for(gather, timeout)

        ## Listen for and process syscalls
        try:
            results = await gather
        except asyncio.TimeoutError:
            warning('Ptrace event timed out')
            raise ChannelTimeoutException("Channel timeout when waiting for syscall")
        return results

    async def close(self):
        if self.root:
            await self.root.terminateTree()
        self._debugger.unsubscribe(self._dbgsub)
        await self._debugger.quit()

    def _del_observed(self):
        """
        This ensures that observed processes do not get terminated by
        debugger.kill_all().
        Responsibility then falls on the caller to transfer the observed to
        another debugger or ultimately terminate them.
        """
        for process in self.observed:
            self._debugger.deleteProcess(process)

    def __del__(self):
        self._del_observed()
        self._debugger.kill_all()

class PtraceForkChannel(PtraceChannel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._proc_trapped = False
        self._proc_untrap = False
        self._wait_for_proc = False
        self._forked_child = None

    async def setup(self):
        await super().setup()
        # extract forkserver location
        self._symb_forkserver = resolve_symbol(self._proc, 'forkserver')

    @property
    def root(self):
        return self._forked_child

    @root.setter
    def root(self, value):
        self._forked_child = value

    async def process_exit(self, event):
        try:
            await super().process_exit(event)
        finally:
            # if the current target exits unexpectedly, we also report it to the forkserver
            if event.process == self._forked_child:
                await self._wakeup_forkserver()
            elif event.process == self._proc:
                error("Forkserver crashed!")
                raise ForkserverCrashedException

    async def process_signal(self, event):
        if event.signum == signal.SIGTRAP:
            # If the forkserver traps, that means it's waiting for its child to
            # die. We will wake it up when we kill the forked_child.
            # Otherwise, the child trapped and we resume its execution
            if event.process != self._proc:
                forked_child = event.process
                self._forked_child = forked_child.root = forked_child
                # restore correct trap byte and registers
                self._cleanup_forkserver(forked_child)
                # resume execution of the child process
                self.resume_process(forked_child)
            else:
                debug("Forkserver trapped, waiting for wake-up call")
                self._proc_trapped = True
                if self._proc_untrap:
                    # When the child dies unexpectedly (ForkChildKilledEvent),
                    # we wake up the server
                    await self._wakeup_forkserver()
                    self._proc_untrap = False
                self._wait_for_proc = False
        elif event.signum == signal.SIGCHLD and event.process == self._proc:
            # when the forkserver receives SIGCHLD, we ignore it
            self.resume_process(event.process)
        else:
            await super().process_signal(event)

    async def process_auxiliary_event(self, event, ignore_callback):
        try:
            raise event
        except NewProcessEvent as event:
            await self.process_new(event, ignore_callback)
            if event.process.parent == self._proc and event.process.is_attached:
                # pause children and queue up syscalls until forkserver traps
                self._wait_for_proc = True
        except ForkChildKilledEvent as event:
            # this is sent by the debugger when the child dies unexpectedly (custom behavior)
            # FIXME check if there are exceptions to this
            # We need to resume the forkserver because addProcess raised an exception instead
            # of NewProcessEvent (so the forkserver is still stuck in the fork syscall)
            warning("Forked child died on entry, forkserver will be woken up")
            self.resume_process(self._proc)
            self._proc_untrap = True
            CountProfiler("infant_mortality")(1)
        except Exception as event:
            await super().process_auxiliary_event(event, ignore_callback)

    async def push_observe(self, *args, **kwargs) -> Optional[PtraceProcess]:
        if self._forked_child:
            await self._wakeup_forkserver()
        return await super().push_observe(*args, **kwargs)

    async def _wait_for_syscall(self, process: PtraceProcess=None, **kwargs):
        event = None
        while not event:
            if self._wait_for_proc and process is not self._proc:
                event = await super()._wait_for_syscall(self._proc, **kwargs)
            else:
                event = await super()._wait_for_syscall(process, **kwargs)

            if process and event.process is not process:
                # we process the event out-of-order
                opts = DefaultOptions
                await self.process_event(event, opts)
                event = None
        return event

    def _stack_push(self, process, value):
        rsp = process.getStackPointer() - 8
        process.writeBytes(rsp, value.to_bytes(8, byteorder=sys.byteorder))
        process.setreg(CPU_STACK_POINTER, rsp)

    def _stack_pop(self, process):
        rsp = process.getStackPointer()
        value = int.from_bytes(process.readBytes(rsp, 8), byteorder=sys.byteorder)
        process.setreg(CPU_STACK_POINTER, rsp + 8)
        return value

    def _inject_forkserver(self, process: PtraceProcess, address: int):
        debug("Injecting forkserver!")
        with process.regsctx():
            self._trap_rip = address # we need this to restore execution later
            word_offset = address % CPU_WORD_SIZE
            self._trap_rip_aligned = address - word_offset
            self._trap_rsp = process.getStackPointer()

            # read the original word where a byte needs be replaced by a trap
            self._trap_asm_word = process.readWord(self._trap_rip_aligned)

            # place a trap
            mask_shift = word_offset * 8
            trap_mask = 0xff << mask_shift
            trap_word = (self._trap_asm_word & ~trap_mask) | (0xCC << mask_shift)
            process.writeWord(self._trap_rip_aligned, trap_word)
            process.setreg(CPU_STACK_POINTER, self._trap_rsp & ~0x0F)

            # set up the stack, so that it returns to the trap
            self._stack_push(process, 0) # some x86-64 alignment stuff
            self._stack_push(process, address)

            # redirect control flow to the forkserver
            process.setreg(CPU_INSTR_POINTER, self._symb_forkserver)

    def _invoke_forkserver(self, process: PtraceProcess):
        address = process.getInstrPointer()
        self._inject_forkserver(process, address)

    def _cleanup_forkserver(self, process: PtraceProcess):
        with process.regsctx():
            # restore the original byte that was replaced by the trap
            process.writeWord(self._trap_rip_aligned, self._trap_asm_word)
            # restore the stack pointer to its proper location
            process.setreg(CPU_STACK_POINTER, self._trap_rsp)
            # redirect control flow back where it should have resumed
            process.setreg(CPU_INSTR_POINTER, self._trap_rip)

    async def _wakeup_forkserver(self):
        if self._proc_trapped:
            debug("Waking up forkserver :)")
            self.resume_process(self._proc)

            if not self._use_seccomp:
                # must actually wait for syscall, not any event
                self._wakeup_forkserver_syscall_found = False

                # backup the old ignore_callbacks
                for process in self._debugger:
                    process.syscall_state._ignore_callback = process.syscall_state.ignore_callback

                await self.monitor_syscalls(None,
                    self._wakeup_forkserver_ignore_callback,
                    self._wakeup_forkserver_break_callback,
                    self._wakeup_forkserver_syscall_callback,
                    break_on_entry=True, break_on_exit=True)
                # restore the old ignore_callbacks
                for process in self._debugger:
                    process.syscall_state.ignore_callback = process.syscall_state._ignore_callback
                    del process.syscall_state._ignore_callback

            self._proc_trapped = False

    async def close(self):
        if self._forked_child:
            await self._forked_child.terminateTree()
        if self._proc_trapped:
            # when we kill the forked_child, we wake up the forkserver from trap
            await self._wakeup_forkserver()

    ### Callbacks ###
    def _invoke_forkserver_ignore_callback(self, syscall):
        return True

    async def _invoke_forkserver_break_callback(self):
        return self._proc_trapped

    async def _invoke_forkserver_syscall_callback(self, process, syscall):
        pass

    def _wakeup_forkserver_ignore_callback(self, syscall):
        return False

    async def _wakeup_forkserver_break_callback(self):
        return self._wakeup_forkserver_syscall_found

    async def _wakeup_forkserver_syscall_callback(self, process, syscall):
        is_entry = syscall.result is None
        self._wakeup_forkserver_syscall_found = True
        if is_entry:
            process.syscall()
        return is_entry

@dataclass(frozen=True)
class PtraceChannelFactory(AbstractChannelFactory,
        capture_paths=('driver.use_seccomp',)):
    use_seccomp: bool = HAS_SECCOMP_FILTER

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['driver'].get('type') == 'unix'

    @abstractmethod
    async def create(self, pobj: Popen, netns: str, **kwargs) -> PtraceChannel:
        pass

@dataclass
class Environment:
    """
    This class describes a process execution environment.
    """
    path: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = None # field(default_factory=dict)
    cwd: str = None
    stdin:  IO = None
    stdout: IO = None
    stderr: IO = None

class ProcessDriver(BaseDriver,
        capture_components={ComponentType.channel_factory},
        capture_paths=(
            'driver.exec', 'driver.disable_aslr', 'fuzzer.work_dir',
            'driver.isolate_fs', 'driver.isolate_net'
            )):
    PROC_TERMINATE_RETRIES = 5
    PROC_TERMINATE_WAIT = 0.1 # seconds

    def __init__(self, *, exec: dict, disable_aslr: bool, work_dir:str,
            isolate_fs: bool=True, isolate_net: bool=True,
            channel_factory: PtraceChannelFactory,
            **kwargs):
        super().__init__(channel_factory=channel_factory, **kwargs)
        self._work_dir = Path(work_dir)
        self._shared_dir = self._work_dir / 'shared'
        self._isolate_fs = isolate_fs
        if isolate_fs:
            if not HAS_NAMESPACES:
                raise RuntimeError("Namespace isolation is not available")
            self._mount_caps = {
                CAP_CHOWN, CAP_DAC_OVERRIDE, CAP_FOWNER, CAP_SETPCAP,
                CAP_SYS_ADMIN}
            if not has_caps(*self._mount_caps):
                warning("Python does not have the required capabilities for"
                        " mounting an isolated fs. Make sure to provide it with"
                        " sufficient capabilities, or disable isolation.")
                self._isolate_fs = False

        self._isolate_net = isolate_net
        if isolate_net:
            if not HAS_NAMESPACES:
                raise RuntimeError("Namespace isolation is not available")
            self._net_caps = {CAP_NET_ADMIN, CAP_SYS_ADMIN, CAP_DAC_OVERRIDE}
            if not has_caps(*self._net_caps):
                warning("Python does not have the required capabilities for a"
                        " network namespace. Make sure to provide it with"
                        " sufficient capabilities, or disable isolation.")
                self._isolate_net = False
        self._exec_env = self.setup_execution_environment(exec)
        self._pobj = None # Popen object of child process
        self._netns_name = f'ns:{uuid4()}'
        self._disable_aslr = disable_aslr

        if disable_aslr:
            ADDR_NO_RANDOMIZE = 0x0040000
            personality = ctypes.pythonapi.personality
            personality.restype = ctypes.c_int
            personality.argtypes = [ctypes.c_ulong]
            personality(ADDR_NO_RANDOMIZE)

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['driver'].get('type') == 'unix'

    def setup_execution_environment(self, config: dict) -> Environment:
        ValueProfiler('target_name')(config["path"])
        for stdf in ["stdin", "stdout", "stderr"]:
            if config.get(stdf) == "inherit":
                config[stdf] = None
            elif config.get(stdf) is not None:
                config[stdf] = open(config[stdf], "wt")
            elif stdf == "stdin":
                config[stdf] = PIPE
            else:
                config[stdf] = DEVNULL
        if not config.get("env"):
            config["env"] = dict(os.environ)
        config["env"]["TANGO_WORKDIR"] = str(self._work_dir)
        config["env"]["TANGO_SHAREDDIR"] = str(self._shared_dir)
        if self._isolate_fs:
            # could be used by the forkserver to clean up instead of remounting
            # FIXME path is re-evaluated here because of how Popen does not
            # allow changing of env within preexec_fn
            fspath = self._work_dir.resolve().relative_to('/') / 'fs'
            tmpfs = Path('/rootfs') / fspath / 'tmpfs'
            upper = tmpfs / 'upper'
            work = tmpfs / 'work'
            lower = Path('/rootfs') / fspath / 'rootfs'
            overlay = Path('/rootfs') / fspath / 'overlayfs'
            config["env"]["TANGO_FS_UPPERDIR"] = str(upper)
            config["env"]["TANGO_FS_WORKDIR"] = str(work)
            config["env"]["TANGO_FS_LOWERDIR"] = str(lower)
            config["env"]["TANGO_FS_OVERLAY"] = str(overlay)

        config["args"][0] = os.path.realpath(config["args"][0])
        if not (path := config.get("path")):
            config["path"] = config["args"][0]
        else:
            config["path"] = os.path.realpath(path)
        if (cwd := config.get("cwd")):
            config["cwd"] = os.path.realpath(cwd)
        return Environment(**config)

    def __del__(self):
        netns.remove(self._netns_name)

    async def relaunch(self):
        ## Kill current process, if any
        if self._pobj:
            # ensure that the channel is closed and the debugger detached
            await self._channel.close()

            # close pipes, if any
            for f in ('in', 'out', 'err'):
                if (stdf := getattr(self._pobj, f'std{f}')):
                    stdf.close()

            retries = 0
            while True:
                if retries == self.PROC_TERMINATE_RETRIES:
                    # TODO add logging to indicate force kill
                    # FIXME is safe termination necessary?
                    self._pobj.kill()
                    break
                self._pobj.terminate()
                try:
                    self._pobj.wait(self.PROC_TERMINATE_WAIT)
                    break
                except TimeoutExpired:
                    retries += 1

        ## Launch new process
        self._pobj = self._popen()

        ## Establish a connection
        await self.create_channel()

    async def create_channel(self, **kwargs):
        self._channel = await self._factory.create(self._pobj, self._netns_name,
            **kwargs)
        return self._channel

    @property
    def channel(self):
        return self._channel

    def _popen(self):
        pobj = Popen(self._exec_env.args, shell=False,
            bufsize=0,
            executable = self._exec_env.path,
            stdin  = self._exec_env.stdin,
            stdout = self._exec_env.stdout,
            stderr = self._exec_env.stderr,
            cwd = self._exec_env.cwd,
            restore_signals = True, # TODO check if this should be false
            env = self._exec_env.env,
            preexec_fn = self._preexec_fn,
            start_new_session = True
        )
        return pobj

    def _preexec_fn(self):
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, {})
            self._setup_container()
            if self._factory.use_seccomp:
                self._install_seccomp_filter()
        except Exception as ex:
            import traceback
            error(ex)
            error(traceback.format_exc())
            raise SystemExit from ex
        ptrace_traceme()

    @staticmethod
    def _install_seccomp_filter():
        if not HAS_SECCOMP_FILTER:
            raise NotImplementedError
        filt = BPF_FILTER(
            BPF_STMT(BPF_LD | BPF_W | BPF_ABS, seccomp_data.nr.offset),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, NR['pselect6'], 17, 0),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, NR['accept'], 16, 0),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, NR['accept4'], 15, 0),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, NR['bind'], 14, 0),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, NR['close'], 13, 0),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, NR['dup'], 12, 0),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, NR['dup2'], 11, 0),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, NR['dup3'], 10, 0),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, NR['listen'], 9, 0),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, NR['poll'], 8, 0),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, NR['ppoll'], 7, 0),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, NR['read'], 6, 0),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, NR['recvfrom'], 5, 0),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, NR['recvmsg'], 4, 0),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, NR['select'], 3, 0),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, NR['shutdown'], 2, 0),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, NR['socket'], 1, 0),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRACE | SECCOMP_RET_DATA),
        )
        prog = BPF_PROG(filt)
        if prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0):
            raise RuntimeError("Failed to set no_new_privs")

        syscall = ctypes.pythonapi.syscall
        syscall.restype = ctypes.c_int
        syscall.argtypes = (ctypes.c_uint64,)*4
        if (res := syscall(NR['seccomp'], SECCOMP_SET_MODE_FILTER,
                SECCOMP_FILTER_FLAG_TSYNC, ctypes.addressof(prog))):
            raise RuntimeError("Failed to install seccomp bpf")

    def _setup_container(self, drop_capabilities=True):
        caps = set()
        if self._isolate_net:
            caps |= self._net_caps
            netns.setns(self._netns_name, flags=os.O_CREAT)
            with IPRoute() as ipr:
                ipr.link('set', index=1, state='up')
        if self._isolate_fs:
            caps |= self._mount_caps
            self._setup_volatile_filesystem(
                self._work_dir / "fs", self._shared_dir)

        if drop_capabilities:
            # finally we drop all acquired caps
            drop_caps(*caps)

    @staticmethod
    def _switch_mount_namespace():
        try:
            unshare(CLONE_NEWNS)
        except Exception as ex:
            raise RuntimeError("Failed to detach mount namespace") \
                from ex
        try:
            mount('rootfs', '/', None, MS_REC | MS_PRIVATE, None)
        except Exception as ex:
            raise RuntimeError("Failed to remount rootfs as private") from ex

    @staticmethod
    def _setup_overlay_tmpfs(path):
        path = path.absolute()
        # first, we setup a read-only rootfs as the base for the overlay
        rootfs = path / 'rootfs'
        if not rootfs.is_mount():
            rootfs.mkdir(parents=True, exist_ok=True)
            try:
                mount('/', rootfs, None, MS_BIND, None)
                mount(None, rootfs, None, MS_REMOUNT|MS_BIND|MS_RDONLY, None)
            except Exception as ex:
                raise RuntimeError("Failed to mount r/o rootfs") from ex

        # next, we setup a tmpfs as the volatile storage for overlayfs
        tmpfs = path / 'tmpfs'
        overlayfs = path / 'overlayfs'
        if tmpfs.is_mount():
            # it could still be used by an existing overlayfs mount
            if overlayfs.is_mount():
                umount(overlayfs)
            umount(tmpfs)
        tmpfs.mkdir(parents=True, exist_ok=True)
        try:
            mount('tmpfs', tmpfs, 'tmpfs', 0, None)
        except Exception as ex:
            raise RuntimeError("Failed to mount tmpfs") from ex
        upperdir = tmpfs / 'upper'
        workdir = tmpfs / 'work'
        upperdir.mkdir(parents=True, exist_ok=True)
        workdir.mkdir(parents=True, exist_ok=True)

        # finally, we mount the overlayfs and return a path to it
        if overlayfs.is_mount():
            umount(overlayfs)
        overlayfs.mkdir(parents=True, exist_ok=True)
        options = f'lowerdir={rootfs},upperdir={upperdir},workdir={workdir}'
        try:
            mount('overlayfs', overlayfs, 'overlay',
                (MS_NOATIME | MS_NODEV | MS_NOSUID) and 0, options)

            # we fix up procfs and devtmpfs in case they are needed
            mount('proc', overlayfs / 'proc', 'proc', 0, None)
            mount('dev', overlayfs / 'dev', 'devtmpfs', 0, None)

            # bind-mount /dev/shm and /run for shmem and netns
            mount('/dev/shm', overlayfs / 'dev/shm', None, MS_BIND, None)
            mount('/run', overlayfs / 'run', None, MS_BIND, None)
            mount(None, overlayfs / 'run', None, MS_REMOUNT|MS_BIND|MS_RDONLY, None)
        except Exception as ex:
            error(ex)
            raise RuntimeError("Failed to mount overlayfs") from ex

        return overlayfs, upperdir

    @classmethod
    def _setup_volatile_filesystem(cls, fs_path, shared_dir):
        cls._switch_mount_namespace()

        # we obtain the cwd before pivoting root
        cwd = Path.cwd()

        new_root, upperdir = cls._setup_overlay_tmpfs(fs_path)

        sharedfs = new_root / shared_dir.absolute().relative_to("/")
        sharedfs.mkdir(parents=True, exist_ok=True)
        try:
            mount(shared_dir, sharedfs, None, MS_BIND, None)
        except Exception as ex:
            raise RuntimeError("Failed to mount shared dir") from ex

        old_root = new_root / "rootfs"
        old_root.mkdir(exist_ok=True)
        try:
            pivot_root(new_root, old_root)
        except Exception as ex:
            raise RuntimeError("Failed to pivot root in isolated namespace") \
                from ex

        # clean up
        # (path / "rootfs").rmdir()
        # path.rmdir()

        # finally we chdir into the cwd within the new fs
        os.chdir(cwd)

class ProcessForkDriver(ProcessDriver):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['driver'].get('forkserver')

    async def relaunch(self):
        if not self._pobj:
            ## Launch new process
            self._pobj = self._popen()
        elif self._channel:
            ## Kill current process, if any
            try:
                await self._channel.close()
            except ProcessLookupError:
                pass

        ## Establish a connection
        await self.create_channel()

class ForkserverCrashedException(RuntimeError):
    pass

class FileDescriptorChannel(PtraceChannel):
    def __init__(self, *, data_timeout: float, **kwargs):
        super().__init__(**kwargs)
        self._refcounter = dict()
        self._data_timeout = data_timeout * self._timescale if data_timeout \
            else None
        self.synced = False

    async def process_new(self, *args, **kwargs):
        await super().process_new(*args, **kwargs)
        for fd in self._refcounter:
            self._refcounter[fd] += 1

    async def monitor_syscalls(self,
            monitor_target: Callable,
            ignore_callback: Callable[[PtraceSyscall], bool],
            break_callback: Callable[..., bool],
            syscall_callback: Callable[[PtraceProcess, PtraceSyscall], None], /,
            timeout: float = None,
            process: PtraceProcess = None,
            **kwargs):
        orig_kwargs = kwargs | {'orig_syscall_callback': syscall_callback,
            'orig_ignore_callback': ignore_callback}
        kwargs['break_on_entry'] = True
        kwargs['break_on_exit'] = True
        new_ignore_callback = partial(
            self._tracer_ignore_callback, ignore_callback)
        new_syscall_callback = self._tracer_syscall_callback
        return await super().monitor_syscalls(
            monitor_target, new_ignore_callback,
            break_callback, new_syscall_callback, timeout=timeout,
            process=process, kw=orig_kwargs, **kwargs)

    @abstractmethod
    async def read_bytes(self) -> ByteString:
        pass

    @abstractmethod
    async def write_bytes(self, data) -> int:
        pass

    @property
    @abstractmethod
    def file(self):
        pass

    async def sync(self):
        async def break_cb():
            return self.synced
        async def syscall_cb(syscall):
            assert False # will never be called
        self.synced = False
        ignore_cb = lambda s: True
        await self.monitor_syscalls(None, ignore_cb, break_cb, syscall_cb,
            break_on_entry=False, break_on_exit=False,
            timeout=self._data_timeout)

    async def send(self, data: ByteString) -> int:
        # flush any data left in the receive buffer
        while await self.receive():
            pass

        if len(data):
            sent = await self._send_some(data)
            await self.sync()
            assert self.synced
        else:
            sent = 0
        debug("Sent data to target: %s", data[:sent])
        return sent

    async def receive(self) -> ByteString:
        if not self.is_file_readable():
            return b''
        return await self.read_bytes()

    def is_file_readable(self):
        try:
            poll, _, _ = select.select([self.file], [], [], 0)
            if self.file not in poll:
                return False
        except ValueError:
            raise ChannelBrokenException("file fd is invalid")
        return True

    async def _send_some(self, data: ByteString) -> int:
        self._send_server_received = 0
        self._send_client_sent = 0
        # Set up a barrier so that client_sent is ready when checking for break
        self._send_barrier = asyncio.Barrier(2)
        self._send_barrier_passed = False

        self._send_data = data
        _, ret = await self.monitor_syscalls(self._send_monitor_target,
            self._send_ignore_callback, self._send_break_callback,
            self._send_syscall_callback, timeout=self._data_timeout)
        return ret

    def dup_callback(self, process: PtraceProcess, syscall: PtraceSyscall):
        self._refcounter[syscall.result] = 1
        debug("File descriptor duplicated in dup_fd=%i", syscall.result)

    def close_callback(self, process: PtraceProcess, syscall: PtraceSyscall):
        if syscall.result != 0:
            return
        fd = syscall.arguments[0].value
        self._refcounter[fd] -= 1
        if not self._refcounter[fd]:
            self._refcounter.pop(fd)
        if sum(self._refcounter.values()) == 0:
            raise ChannelBrokenException(
                "Channel closed while waiting for server to read")

    def sync_callback(self, process: PtraceProcess, syscall: PtraceSyscall):
        self.synced = True

    ## Callbacks
    def _dup_ignore_callback(self, syscall: PtraceSyscall) -> bool:
        return syscall.name not in ('dup', 'dup2', 'dup3')

    def _close_ignore_callback(self, syscall: PtraceSyscall) -> bool:
        return syscall.name != 'close'

    def _select_ignore_callback(self, syscall: PtraceSyscall) -> bool:
        return syscall.name not in ('read', 'poll', 'ppoll', 'select',
            'pselect6')

    def _tracer_ignore_callback(self, orig_ignore_callback, syscall):
        return all((orig_ignore_callback(syscall),
            self._dup_ignore_callback(syscall),
            self._close_ignore_callback(syscall),
            self._select_ignore_callback(syscall)
        ))

    async def _tracer_syscall_callback(self,
            process: PtraceProcess, syscall: PtraceSyscall, *, kw, **kwargs):
        is_entry = syscall.result is None
        processed = False

        orig_ignore_callback = kw['orig_ignore_callback']
        orig_break_on_entry = kw.get('break_on_entry', False)
        orig_break_on_exit = kw.get('break_on_exit', True)
        orig_match_condition = (is_entry and orig_break_on_entry) or \
            (not is_entry and orig_break_on_exit)
        if not orig_ignore_callback(syscall) and orig_match_condition:
            orig_syscall_callback = kw['orig_syscall_callback']
            processed = await orig_syscall_callback(process, syscall, **kwargs)

        if not self._dup_ignore_callback(syscall) and not is_entry:
            self._dup_syscall_exit_callback_internal(process, syscall)
        elif not self._close_ignore_callback(syscall) and not is_entry:
            self._close_syscall_exit_callback_internal(process, syscall)
        elif not self._select_ignore_callback(syscall) and is_entry:
            self._select_syscall_entry_callback_internal(process, syscall)

        if is_entry and not processed:
            process.syscall()
            processed = True
        return processed

    def _dup_syscall_exit_callback_internal(self,
            process: PtraceProcess, syscall: PtraceSyscall):
        if syscall.arguments[0].value not in self._refcounter:
            return
        assert syscall.result >= 0
        self.dup_callback(process, syscall)

    def _close_syscall_exit_callback_internal(self,
            process: PtraceProcess, syscall: PtraceSyscall):
        if syscall.arguments[0].value not in self._refcounter:
            return
        self.close_callback(process, syscall)

    def _select_syscall_entry_callback_internal(self,
            process: PtraceProcess, syscall: PtraceSyscall):
        if not self._refcounter:
            return
        matched_fd = self._select_match_fds(process, syscall, self._refcounter)
        if matched_fd is not None:
            self.sync_callback(process, syscall)

    @classmethod
    def _select_match_fds(cls, process: PtraceProcess, syscall: PtraceSyscall,
            fds: Iterable[int]) -> Optional[int]:
        matched_fd = None
        match syscall.name:
            case 'read':
                if syscall.arguments[0].value not in fds:
                    return None
                matched_fd = syscall.arguments[0].value
            case 'poll' | 'ppoll':
                nfds = syscall.arguments[1].value
                pollfds = syscall.arguments[0].value
                fmt = '@ihh'
                size = struct.calcsize(fmt)
                for i in range(nfds):
                    fd, events, revents = struct.unpack(fmt, process.readBytes(
                        pollfds + i * size, size))
                    if fd in fds and (events & select.POLLIN) != 0:
                        matched_fd = fd
                        args = syscall.argument_values
                        # convert call to blocking
                        if syscall.name == 'poll' and args[2] == 0:
                            args = args[:2] + (-1,) + args[3:]  # inf timeout
                        elif syscall.name == 'ppoll' and args[2] != 0:
                            args = args[:2] + (0,) + args[3:]  # nullptr
                        else:
                            break
                        syscall.writeArgumentValues(*args)
                        break
                else:
                    return None
            case 'select' | 'pselect6':
                nfds = syscall.arguments[0].value
                if nfds <= max(fds):
                    return None
                readfds = syscall.arguments[1].value
                fmt = '@l'
                size = struct.calcsize(fmt)
                for fd in fds:
                    l_idx = fd // (size * 8)
                    b_idx = fd % (size * 8)
                    fd_set, = struct.unpack(fmt, process.readBytes(
                        readfds + l_idx * size, size))
                    if fd_set & (1 << b_idx) != 0:
                        matched_fd = fd
                        args = syscall.argument_values
                        # convert call to blocking
                        if args[4]:
                            args = args[:4] + (0,) + args[5:]  # nullptr
                        else:
                            break
                        syscall.writeArgumentValues(*args)
                        break
                else:
                    return None
            case _:
                return None
        return matched_fd

    def _send_ignore_callback(self, syscall):
        return syscall.name != 'read'

    async def _send_syscall_callback(self, process, syscall):
        if not self._send_ignore_callback(syscall) and \
                syscall.arguments[0].value in self._refcounter:
            if syscall.result <= 0:
                raise ChannelBrokenException(
                    "Target failed to read data off file")
            self._send_server_received += syscall.result

    async def _send_break_callback(self):
        if not self._send_barrier_passed:
            try:
                await self._send_barrier.wait()
            except asyncio.BrokenBarrierError:
                raise ChannelBrokenException("Barrier broke while waiting")
            else:
                self._send_barrier_passed = True
        assert self._send_client_sent > 0
        debug("client_sent=%i; server_received=%i",
            self._send_client_sent, self._send_server_received)
        if self._send_server_received > self._send_client_sent:
            raise ChannelBrokenException("Target received too many bytes!")
        return self._send_server_received == self._send_client_sent

    async def _send_monitor_target(self):
        try:
            ret = await self.write_bytes(self._send_data)
            if ret == 0:
                raise ChannelBrokenException("Failed to send any data")
        except Exception:
            await self._send_barrier.abort()
            raise
        self._send_client_sent = ret
        await self._send_barrier.wait()
        return ret

@dataclass(kw_only=True, frozen=True)
class FileDescriptorChannelFactory(PtraceChannelFactory,
        capture_paths=('channel.data_timeout',)):
    data_timeout: float = None # seconds

class SharedMemoryObject:
    valid_chars = frozenset("-_. %s%s" % (ascii_letters, digits))

    def __init__(self, path, typ_ctor, create=False, force=False):
        # default vals so __del__ doesn't fail if __init__ fails to complete
        self._mem = None
        self._map = None
        self._owner = create
        path = self.ensure_tag(path)

        # get the size of the coverage array
        sz_type = ctypes.c_size_t
        _mem, _map = self.mmap_obj(path,
            ctypes.sizeof(sz_type), False, False)
        self._size = sz_type.from_address(self.address_of_buffer(_map)).value
        _map.close()

        self._type = typ_ctor(self._size)
        assert self._size == ctypes.sizeof(self._type)
        self._mem, self._map = self.mmap_obj(path,
            ctypes.sizeof(sz_type) + self._size, create, force)
        self._obj = self._type.from_address(
            self.address_of_buffer(self._map) + ctypes.sizeof(sz_type))

    @classmethod
    def ensure_tag(cls, tag):
        assert frozenset(tag[1:]).issubset(cls.valid_chars)
        if tag[0] != "/":
            tag = "/%s" % (tag,)
        return tag

    @staticmethod
    def mmap_obj(tag, size, create, force, *, truncate=False):
        # assert 0 <= size < sys.maxint
        assert 0 <= size < sys.maxsize

        if truncate:
            ftrunc_sz = size
        else:
            ftrunc_sz = 0
        flag = posix_ipc.O_CREX if create else 0
        try:
            _mem = posix_ipc.SharedMemory(tag, flags=flag, size=ftrunc_sz)
        except posix_ipc.ExistentialError:
            if force:
                posix_ipc.unlink_shared_memory(tag)
                _mem = posix_ipc.SharedMemory(tag, flags=flag, size=ftrunc_sz)
            else:
                raise

        _map = mmap.mmap(_mem.fd, size)
        _mem.close_fd()

        return _mem, _map

    @staticmethod
    def address_of_buffer(buf):
        return ctypes.addressof(ctypes.c_char.from_buffer(buf))

    def clone_object(self, onto: Optional[ctypes.Array]=None):
        if onto is None:
            onto = self._type()
        ctypes.memmove(onto, self._obj, self._size)
        return onto

    def write_object(self, data: ctypes.Array):
        ctypes.memmove(self._obj, data, self._size)

    @property
    def object(self):
        return self._obj

    @property
    def size(self):
        return self._size

    @property
    def ctype(self):
        return self._type

    @property
    def address(self):
        return self._map

    def __del__(self):
        if self._map is not None:
            self._map.close()
        if self._mem is not None and self._owner:
            self._mem.unlink()

def resolve_dwarf_symbol(elf: ELFFile, symbol: str):
    if not elf.has_dwarf_info():
        raise ValueError("DWARF info is needed for resolving symbols")
    dwarf = elf.get_dwarf_info()
    try:
        debug("Searching for symbol=%s", symbol)
        die = next(die
            for cu in dwarf.iter_CUs()
                for die in cu.iter_DIEs()
                if 'DW_AT_name' in die.attributes and
                    die.attributes['DW_AT_name'].value == symbol.encode()
        )
    except StopIteration:
        raise KeyError(f"{symbol=} not found")
    rel_addr = die.attributes['DW_AT_low_pc'].value
    return rel_addr

def resolve_symtab_symbol(elf: ELFFile, symbol: str):
    symtab = elf.get_section_by_name('.symtab')
    if not symtab:
        raise ValueError(
            "An unstripped binary must be used for resolving symbols!")
    symbols = symtab.get_symbol_by_name(symbol)
    if not symbols:
        raise KeyError(f"{symbol=} not found")
    elif len(symbols) > 1:
        raise ValueError("Multiple symbols found, ambiguous resolution")

    symbol, = symbols
    rel_addr = symbol['st_value']
    return rel_addr

def resolve_symbol(process: PtraceProcess, symbol: str):
    path = os.path.realpath(readProcessStat(process.pid).program)
    with open(path, 'rb') as file:
        elf = ELFFile(file)
        for fn in (resolve_symtab_symbol, resolve_dwarf_symbol):
            try:
                rel_addr = fn(elf, symbol)
                break
            except (KeyError, ValueError):
                pass
        else:
            raise RuntimeError(f"Failed to find {symbol=}")

        if elf.header['e_type'] == 'ET_EXEC':
            addr = rel_addr
        elif elf.header['e_type'] == 'ET_DYN':
            # get runtime image base
            vmmaps = process.readMappings()
            try:
                base_map = next(m for m in vmmaps if m.pathname == path)
            except StopIteration:
                raise RuntimeError(
                    "Could not find matching vmmap entry."
                    " Maybe you are using a symbolic link"
                    " as the path to the target?")
            load_base = base_map.start
            addr = rel_addr + load_base
        else:
            raise RuntimeError(f"Entry `{elf.header['e_type']}` not supported")

        debug("symbol=%s found at 0x%016X", symbol, addr)
        return addr
