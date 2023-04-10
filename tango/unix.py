from __future__ import annotations

from . import debug, info, warning, critical, error

from tango.core import (BaseDriver, AbstractState, Transition,
    LoadableTarget, ValueProfiler, CountProfiler, AbstractChannel,
    AbstractChannelFactory)
from tango.ptrace.binding import ptrace_traceme, HAS_SECCOMP_FILTER
from tango.ptrace.cpu_info import CPU_WORD_SIZE
from tango.ptrace.binding.cpu import CPU_INSTR_POINTER, CPU_STACK_POINTER
from tango.ptrace.debugger import   (PtraceDebugger, PtraceProcess,
    ProcessEvent, ProcessExit, ProcessSignal, NewProcessEvent, ProcessExecution,
    ForkChildKilledEvent)
from tango.ptrace import PtraceError
from tango.ptrace.func_call import FunctionCallOptions
from tango.ptrace.syscall import PtraceSyscall, SOCKET_SYSCALL_NAMES
from tango.ptrace.tools import signal_to_exitcode
from tango.common import sync_to_async, GLOBAL_ASYNC_EXECUTOR, ComponentType
from tango.exceptions import (ChannelTimeoutException, StabilityException,
    ProcessCrashedException, ProcessTerminatedException, ChannelBrokenException)

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import IO, AsyncGenerator, Callable, Iterable, Optional
from pyroute2 import netns, IPRoute
from subprocess import Popen, TimeoutExpired, DEVNULL, PIPE
from concurrent.futures import ThreadPoolExecutor
from elftools.elf.elffile import ELFFile
from threading import Event, Thread
from uuid import uuid4
from string import ascii_letters, digits
import os
import sys
import ctypes
import signal
import traceback
import struct
import select
import posix_ipc
import mmap

__all__ = [
    'ProcessDriver', 'ProcessForkDriver',
    'PtraceChannel', 'PtraceForkChannel', 'PtraceChannelFactory',
    'FileDescriptorChannel', 'FileDescriptorChannelFactory',
    'SharedMemoryObject'
]

SOCKET_SYSCALL_NAMES = SOCKET_SYSCALL_NAMES.union(('read', 'write'))

if HAS_SECCOMP_FILTER:
    from tango.ptrace.binding import (BPF_LD, BPF_W, BPF_ABS, BPF_JMP, BPF_JEQ,
        BPF_K, BPF_RET, BPF_STMT, BPF_JUMP, BPF_PROG, BPF_FILTER,
        SECCOMP_RET_ALLOW, SECCOMP_RET_TRACE, SECCOMP_RET_DATA,
        SECCOMP_FILTER_FLAG_TSYNC, SECCOMP_SET_MODE_FILTER)
    from tango.ptrace.binding.linux_struct import seccomp_data
    from tango.ptrace.syscall import SYSCALL_NUMBERS as NR

class PtraceChannel(AbstractChannel):
    def __init__(self, pobj: Popen, *, use_seccomp: bool, **kwargs):
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
        self._proc = self._debugger.addProcess(self._pobj.pid, is_attached=True)

        # FIXME this is never really used; it's just a placeholder that went
        # obsolete
        default_ignore = lambda syscall: syscall.name not in SOCKET_SYSCALL_NAMES
        self.prepare_process(self._proc, default_ignore, resume=True)

        self._monitor_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='PtraceMonitorExecutor')

    def prepare_process(self, process, ignore_callback, resume=True):
        if not process in self._debugger:
            self._debugger.addProcess(process, is_attached=True)

        if self._process_all:
            ignore_callback = lambda x: False
        process.syscall_state.ignore_callback = ignore_callback

        if resume:
            self.resume_process(process)

    def process_syscall(self, process, syscall, syscall_callback, is_entry, **kwargs) -> bool:
        # ensure that the syscall has finished successfully before callback
        if is_entry or syscall.result >= 0:
            # calling syscall.format() takes a lot of time and should be
            # avoided in production, even if logging is disabled
            if self._process_all or self._verbose:
                debug(f"syscall requested: [{process.pid}] {syscall.format()}")
            else:
                debug(f"syscall requested: [{process.pid}] {syscall.name}")
            processed = syscall_callback(process, syscall, **kwargs)
        else:
            processed = False
        return processed

    def process_exit(self, event):
        debug(f"Process with {event.process.pid=} exited, deleting from debugger")
        debug(f"Reason: {event}")
        self._debugger.deleteProcess(event.process)
        if event.exitcode == 0:
            raise ProcessTerminatedException(f"Process with {event.process.pid=} exited normally", exitcode=0)
        elif event.signum is not None:
            raise ProcessCrashedException(f"Process with {event.process.pid=} crashed with {event.signum=}", signum=event.signum)
        else:
            raise ProcessTerminatedException(f"Process with {event.process.pid=} terminated abnormally with {event.exitcode=}", exitcode=event.exitcode)

    def process_signal(self, event):
        if event.signum == signal.SIGUSR2:
            raise ChannelTimeoutException("Channel timeout when waiting for syscall")
        elif event.signum in (signal.SIGINT, signal.SIGWINCH):
            debug(f"Process with {event.process.pid=} received SIGINT or SIGWINCH {event.signum=}")
            # Ctrl-C or resizing the window should not be passed to child
            self.resume_process(event.process)
            return
        elif event.signum == signal.SIGSTOP:
            critical(f"{event.process.pid=} received rogue SIGSTOP, resuming for now")
            self.resume_process(event.process)
            return
        elif event.signum == signal.SIGSEGV:
            raise ProcessCrashedException(f"Process with {event.process.pid=} terminated abnormally with {event.signum=}", exitcode=event.signum)
        debug(f"Target process with {event.process.pid=} received signal with {event.signum=}")
        event.display(log=warning)
        signum = event.signum
        self.resume_process(event.process, signum)
        exitcode = signal_to_exitcode(event.signum)

    def process_new(self, event, ignore_callback):
        # monitor child for syscalls as well. may be needed for multi-thread or multi-process targets
        debug(f"Target process with {event.process.parent.pid=} forked, adding child process with {event.process.pid=} to debugger")
        if event.process.is_attached:
            # sometimes, child process might have been killed at creation,
            # so the debugger detaches it; we check for that here
            self.prepare_process(event.process, ignore_callback, resume=True)
        self.resume_process(event.process.parent)

    def process_exec(self, event):
        debug(f"Target process with {event.process.pid=} called exec; removing from debugger")
        event.process.detach()

    def process_auxiliary_event(self, event, ignore_callback):
        try:
            raise event
        except ProcessExit as event:
            self.process_exit(event)
        except ProcessSignal as event:
            self.process_signal(event)
        except NewProcessEvent as event:
            self.process_new(event, ignore_callback)
        except ProcessExecution as event:
            self.process_exec(event)

    def process_event(self, event, ignore_callback, syscall_callback, *,
        break_on_entry: bool, break_on_exit: bool, **kwargs):
        if event is None:
            return

        is_syscall = event.is_syscall_stop()
        if not is_syscall:
            self.process_auxiliary_event(event, ignore_callback)
        else:
            # Process syscall enter or exit
            # debug(f"Target process with {event.process.pid=} requested a syscall")
            state = event.process.syscall_state

            ### DEBUG ###
            # sc = PtraceSyscall(event.process, self._syscall_options, event.process.getregs())
            # debug(f"syscall traced: [{event.process.pid}] {sc.name=} with {state.name=} and {state.next_event=}")
            #############

            syscall = state.event(self._syscall_options)
            is_entry = state.next_event == 'exit'
            is_exit = not is_entry

            match_condition = (break_on_entry and is_entry) or \
                (break_on_exit and is_exit)
            ignore_condition = syscall is not None and \
                not (self._process_all and ignore_callback(syscall))
            should_process = ignore_condition and match_condition

            processed = False
            try:
                if should_process:
                    processed = self.process_syscall(event.process, syscall,
                        syscall_callback, is_entry, **kwargs)
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

    def timeout_handler(self, stop_event):
        if not stop_event.wait(timeout * self._timescale):
            warning('Ptrace event timed out')
            # send timeout signal to be intercepted by ptrace
            try:
                # get any child process, fail gracefully if all are dead
                proc = next(iter(self._debugger))
                proc.kill(signal.SIGUSR2)
            except Exception as ex:
                # FIXME may still need to signal the debugger somehow that
                # it needs to stop waiting
                warning(traceback.format_exc())

    def _wait_for_syscall(self, process: PtraceProcess=None):
        return self._debugger.waitSyscall(process=process)

    def _monitor_syscalls_internal_loop(self,
            stop_event: Event,
            ignore_callback: Callable[[PtraceSyscall], bool],
            break_callback: Callable[..., bool],
            syscall_callback: Callable[[PtraceProcess, PtraceSyscall], None],
            break_on_entry: bool = False, break_on_exit: bool = True,
            process: PtraceProcess = None,
            **kwargs):
        last_process = None
        while True:
            if not self._debugger:
                raise ProcessTerminatedException("Process was terminated while waiting for syscalls", exitcode=None)

            try:
                debug("Waiting for syscall...")
                # if waitSyscall does not raise an exception, then event is
                # a syscall, otherwise it's some other ProcessEvent
                event = self._wait_for_syscall(process)
                if event is None:
                    continue
                sc = self.process_event(event, ignore_callback, syscall_callback,
                    break_on_entry=break_on_entry, break_on_exit=break_on_exit,
                    **kwargs)
                if sc is not None:
                    last_process = event.process
            except ProcessEvent as e:
                self.process_event(e, ignore_callback, syscall_callback,
                    break_on_entry=break_on_entry, break_on_exit=break_on_exit,
                    **kwargs)

            if break_callback():
                debug("Syscall monitoring finished, breaking out of debug loop")
                if stop_event:
                    stop_event.set()
                break
        return last_process

    def monitor_syscalls(self,
            monitor_target: Callable,
            ignore_callback: Callable[[PtraceSyscall], bool],
            break_callback: Callable[..., bool],
            syscall_callback: Callable[[PtraceProcess, PtraceSyscall], None], /,
            timeout: float = None,
            process: PtraceProcess = None,
            **kwargs):
        procs = (process,) if process else self._debugger
        for proc in procs:
            # update the ignore_callback of processes in the debugger
            if self._process_all:
                ignore_callback = lambda x: False
            proc.syscall_state.ignore_callback = ignore_callback

        ## Execute monitor target
        if monitor_target:
            future = self._monitor_executor.submit(monitor_target)

        ## Listen for and process syscalls
        stop_event = None
        if timeout is not None:
            stop_event = Event()
            timeout_timer = Thread(target=self.timeout_handler, args=(stop_event,))
            timeout_timer.daemon = True
            timeout_timer.start()

        last_process = self._monitor_syscalls_internal_loop(stop_event,
                ignore_callback, break_callback, syscall_callback,
                process=process, **kwargs)

        ## Return the target's result
        result = None
        if monitor_target:
            result = future.result()
        return (last_process, result)

    def terminator(self, process):
        try:
            while True:
                try:
                    # WARN it seems necessary to wait for the child to exit, otherwise
                    # the forkserver may misbehave, and the fuzzer will receive a lot of
                    # ForkChildKilledEvents
                    process.terminate()
                    break
                except PtraceError as ex:
                    critical(f"Attempted to terminate non-existent process ({ex})")
                except ProcessExit as ex:
                    debug(f"{ex.process} exited while terminating {process}")
                    continue
        finally:
            if process in self._debugger:
                self._debugger.deleteProcess(process)
        for p in process.children:
            self.terminator(p)

    def close(self, terminate, **kwargs):
        self._monitor_executor.shutdown(wait=True)
        if terminate:
            self.terminator(self._proc)

    def __del__(self):
        self.close(terminate=True)
        self._debugger.quit()

class PtraceForkChannel(PtraceChannel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._proc_trapped = False
        self._proc_untrap = False
        self._wait_for_proc = False
        self._event_queue = []
        self._forked_child = None

        # extract forkserver location
        with open(self._pobj.args[0], 'rb') as f:
            elf = ELFFile(f)

            # get forkserver offset from image base
            if not elf.has_dwarf_info():
                raise RuntimeError("Debug symbols are needed for forkserver")
            dwarf = elf.get_dwarf_info()
            try:
                debug("Searching for forkserver symbol")
                die = next(die for cu in dwarf.iter_CUs()
                                 for die in cu.iter_DIEs()
                                     if 'DW_AT_name' in die.attributes and
                                         die.attributes['DW_AT_name'].value == b'forkserver'
                )
            except StopIteration:
                raise RuntimeError("Forkserver symbol not found")
            base_addr = die.attributes['DW_AT_low_pc'].value

            if elf.header['e_type'] == 'ET_EXEC':
                self._forkserver = base_addr
            elif elf.header['e_type'] == 'ET_DYN':
                # get runtime image base
                vmmaps = self._proc.readMappings()
                assert vmmaps[0].pathname == self._pobj.args[0], \
                    ("Path to first module in the vmmap did not match the path"
                     " to the executable. Maybe you are using a symbolic link"
                     " as the path to the target?")
                run_base = vmmaps[0].start
                self._forkserver = base_addr + run_base
            debug(f"Forkserver found at 0x{self._forkserver:016X}")

    def process_exit(self, event):
        try:
            super().process_exit(event)
        finally:
            # if the current target exits unexpectedly, we also report it to the forkserver
            if event.process == self._forked_child:
                self._wakeup_forkserver()
            elif event.process == self._proc:
                error("Forkserver crashed!")
                raise ForkserverCrashedException

    def process_signal(self, event):
        if event.signum == signal.SIGTRAP:
            # If the forkserver traps, that means it's waiting for its child to
            # die. We will wake it up when we kill the forked_child.
            # Otherwise, the child trapped and we resume its execution
            if event.process != self._proc:
                self._forked_child = event.process
                # restore correct trap byte and registers
                self._cleanup_forkserver(event.process)
                # resume execution of the child process
                self.resume_process(event.process)
            else:
                debug("Forkserver trapped, waiting for wake-up call")
                self._proc_trapped = True
                if self._proc_untrap:
                    # When the child dies unexpectedly (ForkChildKilledEvent),
                    # we wake up the server
                    self._wakeup_forkserver()
                    self._proc_untrap = False
                self._wait_for_proc = False
        elif event.signum == signal.SIGCHLD and event.process == self._proc:
            # when the forkserver receives SIGCHLD, we ignore it
            self.resume_process(event.process)
        else:
            super().process_signal(event)

    def process_auxiliary_event(self, event, ignore_callback):
        try:
            raise event
        except NewProcessEvent as event:
            self.process_new(event, ignore_callback)
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
            super().process_auxiliary_event(event, ignore_callback)

    def _wait_for_syscall(self, process: PtraceProcess=None):
        # this next block ensures that a forked child does not exit before
        # the forkserver traps. in that scenario, the wake-up call is sent
        # before the forkserver traps, and then it traps forever
        event = None
        if not self._wait_for_proc and self._event_queue:
            event = self._event_queue.pop(0)
        else:
            event = self._debugger.waitSyscall(process=process)
            if self._wait_for_proc and event.process != self._proc:
                self._event_queue.append(event)
                debug("Received event while waiting for forkserver; enqueued!")
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
            process.setreg(CPU_INSTR_POINTER, self._forkserver)

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

    def _wakeup_forkserver(self):
        if self._proc_trapped:
            debug("Waking up forkserver :)")
            self.resume_process(self._proc)

            if not self._use_seccomp:
                # must actually wait for syscall, not any event
                self._wakeup_forkserver_syscall_found = False

                # backup the old ignore_callbacks
                for process in self._debugger:
                    process.syscall_state._ignore_callback = process.syscall_state.ignore_callback

                self.monitor_syscalls(None,
                    self._wakeup_forkserver_ignore_callback,
                    self._wakeup_forkserver_break_callback,
                    self._wakeup_forkserver_syscall_callback,
                    break_on_entry=True, break_on_exit=True)
                # restore the old ignore_callbacks
                for process in self._debugger:
                    process.syscall_state.ignore_callback = process.syscall_state._ignore_callback
                    del process.syscall_state._ignore_callback

            self._proc_trapped = False

    def close(self, terminate, **kwargs):
        if self._forked_child:
            if terminate:
                self.terminator(self._forked_child)
            # when we kill the forked_child, we wake up the forkserver from the trap
            self._wakeup_forkserver()

    ### Callbacks ###
    def _invoke_forkserver_ignore_callback(self, syscall):
        return True

    def _invoke_forkserver_break_callback(self):
        return self._proc_trapped

    def _invoke_forkserver_syscall_callback(self, process, syscall):
        pass

    def _wakeup_forkserver_ignore_callback(self, syscall):
        return False

    def _wakeup_forkserver_break_callback(self):
        return self._wakeup_forkserver_syscall_found

    def _wakeup_forkserver_syscall_callback(self, process, syscall):
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
    def create(self, pobj: Popen, netns: str) -> PtraceChannel:
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
        capture_paths=['driver.exec', 'driver.disable_aslr', 'fuzzer.work_dir']):
    PROC_TERMINATE_RETRIES = 5
    PROC_TERMINATE_WAIT = 0.1 # seconds

    def __init__(self, *, exec: dict, disable_aslr: bool, work_dir:str,
            channel_factory: PtraceChannelFactory, **kwargs):
        super().__init__(channel_factory=channel_factory, **kwargs)
        self._work_dir = work_dir
        self._exec_env = self.setup_execution_environment(exec)
        self._pobj = None # Popen object of child process
        self._netns_name = f'ns:{uuid4()}'

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
        config["env"]["TANGO_WORKDIR"] = self._work_dir
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

    def _prepare_process(self):
        os.setsid()
        netns.setns(self._netns_name, flags=os.O_CREAT)
        with IPRoute() as ipr:
            ipr.link('set', index=1, state='up')
        ptrace_traceme()
        if self._factory.use_seccomp:
            self._install_seccomp_filter()

    @staticmethod
    def _install_seccomp_filter():
        if not HAS_SECCOMP_FILTER:
            raise NotImplementedError
        filt = BPF_FILTER(
            BPF_STMT(BPF_LD | BPF_W | BPF_ABS, seccomp_data.nr.offset),
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
        prctl = ctypes.pythonapi.prctl
        prctl.restype = ctypes.c_int
        prctl.argtypes = (ctypes.c_int,
            ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong
        )
        PR_SET_NO_NEW_PRIVS = 38
        if prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0):
            raise RuntimeError("Failed to set no_new_privs")

        syscall = ctypes.pythonapi.syscall
        syscall.restype = ctypes.c_int
        syscall.argtypes = (ctypes.c_uint64,)*4
        if (res := syscall(NR['seccomp'], SECCOMP_SET_MODE_FILTER,
                SECCOMP_FILTER_FLAG_TSYNC, ctypes.addressof(prog))):
            raise RuntimeError("Failed to install seccomp bpf")

    @sync_to_async(executor=GLOBAL_ASYNC_EXECUTOR)
    def relaunch(self):
        ## Kill current process, if any
        if self._pobj:
            # ensure that the channel is closed and the debugger detached
            self._channel.close(terminate=True)

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
        self._channel = self._factory.create(self._pobj, self._netns_name)

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
            preexec_fn = self._prepare_process
        )
        return pobj

class ProcessForkDriver(ProcessDriver):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['driver'].get('forkserver')

    @sync_to_async(executor=GLOBAL_ASYNC_EXECUTOR)
    def relaunch(self):
        if not self._pobj:
            ## Launch new process
            self._pobj = self._popen()
        elif self._channel:
            ## Kill current process, if any
            try:
                self._channel.close(terminate=True)
            except ProcessLookupError:
                pass

        ## Establish a connection
        self._channel = self._factory.create(self._pobj, self._netns_name)

class ForkserverCrashedException(RuntimeError):
    pass

class FileDescriptorChannel(PtraceChannel):
    def __init__(self, *, data_timeout: float, **kwargs):
        super().__init__(**kwargs)
        self._refcounter = dict()
        self._data_timeout = data_timeout * self._timescale if data_timeout \
            else None
        self.synced = False

    def process_new(self, *args, **kwargs):
        super().process_new(*args, **kwargs)
        for fd in self._refcounter:
            self._refcounter[fd] += 1

    def monitor_syscalls(self,
            monitor_target: Callable,
            ignore_callback: Callable[[PtraceSyscall], bool],
            break_callback: Callable[..., bool],
            syscall_callback: Callable[[PtraceProcess, PtraceSyscall], None], /,
            timeout: float = None,
            process: PtraceProcess = None,
            **kwargs):
        new_ignore_callback = lambda s: all((ignore_callback(s),
            self._dup_ignore_callback(s),
            self._close_ignore_callback(s),
            self._select_ignore_callback(s)
        ))
        orig_kwargs = kwargs | {'orig_syscall_callback': syscall_callback,
            'orig_ignore_callback': ignore_callback}
        kwargs['break_on_entry'] = True
        kwargs['break_on_exit'] = True
        new_syscall_callback = self._tracer_syscall_callback
        return super().monitor_syscalls(monitor_target, new_ignore_callback,
            break_callback, new_syscall_callback, timeout=timeout,
            process=process, kw=orig_kwargs, **kwargs)

    def sync(self):
        self.synced = False
        ignore_cb = lambda s: True
        break_cb = lambda: self.synced
        syscall_cb = lambda p, s: False # will never be called
        self.monitor_syscalls(None, ignore_cb, break_cb, syscall_cb,
            break_on_entry=False, break_on_exit=False,
            timeout=self._data_timeout)

    def dup_callback(self, process: PtraceProcess, syscall: PtraceSyscall):
        self._refcounter[syscall.result] = 1
        debug(f"File descriptor duplicated in dup_fd={syscall.result}")

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

    def _tracer_syscall_callback(self,
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
            processed = orig_syscall_callback(process, syscall, **kwargs)

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
        self.close_callback(process, syscall)

    def _select_syscall_entry_callback_internal(self,
            process: PtraceProcess, syscall: PtraceSyscall):
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
                        args = list(syscall.readArgumentValues(
                            process.getregs()))
                        # convert call to blocking
                        args[2] = -1
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
                        args = list(syscall.readArgumentValues(
                            process.getregs()))
                        # convert call to blocking
                        args[4] = 0
                        syscall.writeArgumentValues(*args)
                        break
                else:
                    return None
            case _:
                return None
        return matched_fd

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
