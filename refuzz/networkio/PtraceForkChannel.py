from . import debug, warning, info

from abc import abstractmethod
from typing import Callable
from networkio import ChannelBase
from   common      import (ChannelBrokenException,
                          ChannelSetupException,
                          ChannelTimeoutException,
                          ProcessCrashedException)
from ptrace import PtraceError
from ptrace.debugger import   (PtraceDebugger,
                               PtraceProcess,
                               ProcessEvent,
                               ProcessExit,
                               ProcessSignal,
                               NewProcessEvent,
                               ProcessExecution,
                               ForkChildKilledEvent)
from ptrace.func_call import FunctionCallOptions
from ptrace.syscall   import PtraceSyscall, SOCKET_SYSCALL_NAMES
from ptrace.tools import signal_to_exitcode
from ptrace.binding.cpu import CPU_INSTR_POINTER, CPU_STACK_POINTER
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread
from subprocess import Popen
import signal
import traceback
import sys

from elftools.elf.elffile import ELFFile

SOCKET_SYSCALL_NAMES = SOCKET_SYSCALL_NAMES.union(('read', 'write'))

class PtraceForkChannel(ChannelBase):
    def __init__(self, pobj: Popen, timescale: float):
        super().__init__(pobj, timescale)

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
        self._syscall_whitelist = SOCKET_SYSCALL_NAMES

        self._debugger = PtraceDebugger()
        self._debugger.traceFork()
        self._proc = self._debugger.addProcess(self._pobj.pid, is_attached=True)
        self._proc_trapped = False
        self._proc_untrap = False
        self._wait_for_proc = False
        self._event_queue = []
        self._prepare_process(self._proc, self._ignore_callback, syscall=True)

        # extract forkserver location
        with open(pobj.args[0], 'rb') as f:
            elf = ELFFile(f)

            # get forkserver offset from image base
            if not elf.has_dwarf_info():
                raise RuntimeError("Debug symbols are needed for forkserver")
            dwarf = elf.get_dwarf_info()
            try:
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
                assert vmmaps[0].pathname == pobj.args[0]
                run_base = vmmaps[0].start
                self._forkserver = base_addr + run_base

    def _ignore_callback(self, syscall):
        return syscall.name not in self._syscall_whitelist

    def _prepare_process(self, process, ignore_callback, syscall=True):
        if not process in self._debugger:
            self._debugger.addProcess(process, is_attached=True)

        process.syscall_state.ignore_callback = ignore_callback
        if syscall:
            process.syscall()

    def _monitor_syscalls(self,
                       monitor_target: Callable,
                       ignore_callback: Callable[[PtraceSyscall], bool],
                       break_callback: Callable[..., bool],
                       syscall_callback: Callable[[PtraceProcess, PtraceSyscall], None],
                       break_on_entry: bool = False,
                       timeout: float = None,
                       **kwargs):

        def process_syscall(process, syscall):
            # ensure that the syscall has finished successfully before callback
            if syscall and syscall.result != -1 and \
                    (break_on_entry or syscall.result is not None):
                syscall_callback(process, syscall, **kwargs)
                debug(f"syscall processed: [{process.pid}] {syscall.format()}")

        def process_auxiliary_event(event):
            try:
                raise event
            except ProcessExit as event:
                exitcode = -256
                if event.exitcode is not None:
                    exitcode = event.exitcode
                warning(f"Process with {event.process.pid=} exited, deleting from debugger")
                warning(f"Reason: {event}")
                self._debugger.deleteProcess(event.process)
                # if the current target exits unexpectedly, we also report it to the forkserver
                if event.process == self.current_target:
                    self._wakeup_forkserver()
                if exitcode == 0:
                    raise ChannelBrokenException(f"Process with {event.process.pid=} exited normally")
                elif self.current_target and event.process == self.current_target:
                    raise ProcessCrashedException(f"Process with {event.process.pid=} crashed with code {exitcode}")
            except ProcessSignal as event:
                if event.signum == signal.SIGUSR2:
                    raise ChannelTimeoutException("Channel timeout when waiting for syscall")
                elif event.signum == signal.SIGTRAP:
                    # if the forkserver traps, that means it's waiting for its child to die
                    # we will wake it up when we kill the current_target
                    # otherwise, the child trapped and we resume its execution
                    if event.process != self._proc:
                        # restore correct trap byte and registers
                        self._cleanup_forkserver(event.process)
                        event.process.syscall()
                    else:
                        debug("Forkserver trapped, waiting for wake-up call")
                        self._proc_trapped = True
                        if self._proc_untrap:
                            # When the child dies unexpectedly (ForkChildKilledEvent), we wake up the server
                            self._wakeup_forkserver()
                            self._proc_untrap = False
                        self._wait_for_proc = False
                    return
                elif event.signum in (signal.SIGINT, signal.SIGWINCH):
                    # Ctrl-C or resizing the window should not be passed to child
                    event.process.syscall()
                    return
                elif event.signum == signal.SIGSTOP:
                    critical(f"{event.process.pid=} received rogue SIGSTOP, resuming for now")
                    event.process.syscall()
                    return
                elif event.signum == signal.SIGCHLD and event.process == self._proc:
                    # when the forkserver receives SIGCHLD, we ignore it
                    event.process.syscall()
                    return
                debug(f"Target process with {event.process.pid=} received signal with {event.signum=}")
                event.display(log=warning)
                signum = event.signum
                event.process.syscall(signum)
                exitcode = signal_to_exitcode(event.signum)
                return
            except NewProcessEvent as event:
                # monitor child for syscalls as well. may be needed for multi-thread or multi-process targets
                debug(f"Target process with {event.process.parent.pid=} forked, adding child process with {event.process.pid=} to debugger")
                if event.process.is_attached:
                    # sometimes, child process might have been killed at creation,
                    # so the debugger detaches it; we check for that here
                    self._prepare_process(event.process, ignore_callback, syscall=True)
                    # pause children and queue up syscalls until forkserver traps
                    self._wait_for_proc = True
                event.process.parent.syscall()
                return
            except ProcessExecution as event:
                debug(f"Target process with {event.process.pid=} called exec; removing from debugger")
                self._debugger.deleteProcess(event.process)
                return
            except ForkChildKilledEvent as event:
                # this is sent by the debugger when the child dies unexpectedly (custom behavior)
                # FIXME check if there are exceptions to this
                # We need to resume the forkserver because addProcess raised an exception instead
                # of NewProcessEvent (so the forkserver is still stuck in the fork syscall)
                warning("Forked child died on entry, forkserver will be woken up")
                self._proc.syscall()
                self._proc_untrap = True

        def process_event(event):
            if event is None:
                return
            syscall_signum = signal.SIGTRAP
            if self._debugger.use_sysgood:
                syscall_signum |= 0x80
            is_syscall = isinstance(event, ProcessSignal) and event.signum == syscall_signum

            if not is_syscall:
                try:
                    raise event
                except ProcessEvent as event:
                    process_auxiliary_event(event)
            else:
                # Process syscall enter or exit
                # debug(f"Target process with {event.process.pid=} requested a syscall")
                state = event.process.syscall_state
                syscall = state.event(self._syscall_options)
                process_syscall(event.process, syscall)
                event.process.syscall()
                return syscall

        def timeout_handler(event):
            if not event.wait(timeout * self._timescale):
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

        with ThreadPoolExecutor() as executor:
            for process in self._debugger:
                # update the ignore_callback of processes in the debugger
                self._prepare_process(process, ignore_callback, syscall=False)

            ## Execute monitor target
            if monitor_target:
                future = executor.submit(monitor_target)

            ## Listen for and process syscalls
            break_event = Event()
            if timeout is not None:
                timeout_timer = Thread(target=timeout_handler, args=(break_event,))
                timeout_timer.daemon = True
                timeout_timer.start()

            while True:
                if not self._debugger:
                    raise ChannelBrokenException("Process was terminated while waiting for syscalls")

                last_process = None
                try:
                    debug("Waiting for syscall...")
                    # this next block ensures that a forked child does not exit before
                    # the forkserver traps. in that scenario, the wake-up call is sent
                    # before the forkserver traps, and then it traps forever
                    if not self._wait_for_proc and self._event_queue:
                        event = self._event_queue.pop(0)
                    else:
                        event = self._debugger.waitSyscall()
                        if self._wait_for_proc and event.process != self._proc:
                            self._event_queue.append(event)
                            continue
                    # even if it is a syscall, we raise it to pass it to process_event()
                    raise event
                except ProcessEvent as e:
                    syscall = process_event(e)
                    if syscall is None:
                        continue
                    last_process = e.process

                if break_callback():
                    debug("Syscall monitoring finished, breaking out of debug loop")
                    break_event.set()
                    break

            ## Return the target's result
            if monitor_target:
                return last_process, future.result()
            else:
                return last_process

    def _stack_push(self, process, value):
        rsp = process.getStackPointer() - 8
        process.writeBytes(rsp, value.to_bytes(8, byteorder=sys.byteorder))
        process.setreg(CPU_STACK_POINTER, rsp)

    def _stack_pop(self, process):
        value = process.readBytes(rsp, 8)
        rsp = process.getStackPointer() + 8
        process.setreg(CPU_STACK_POINTER, rsp)
        return value

    def _invoke_forkserver(self, process: PtraceProcess):
        self._trap_rip = process.getInstrPointer()
        self._trap_rsp = process.getStackPointer()

        # read the original byte to be replaced by a trap
        self._trap_asm = process.readBytes(self._trap_rip, 1)

        # place a trap
        process.writeBytes(self._trap_rip, b'\xCC')
        process.setreg(CPU_STACK_POINTER, self._trap_rsp & ~0x0F)

        # set up the stack, so that it returns to the trap
        self._stack_push(process, 0) # some x86-64 alignment stuff
        self._stack_push(process, self._trap_rip)

        # redirect control flow to the forkserver
        process.setreg(CPU_INSTR_POINTER, self._forkserver)

    def _cleanup_forkserver(self, process: PtraceProcess):
        # restore the original byte that was replaced by the trap
        process.writeBytes(self._trap_rip, self._trap_asm)
        # restore the stack pointer to its proper location
        process.setreg(CPU_STACK_POINTER, self._trap_rsp)
        # redirect control flow back where it should have resumed
        process.setreg(CPU_INSTR_POINTER, self._trap_rip)

    def _wakeup_forkserver(self):
        if self._proc_trapped:
            debug("Waking up forkserver :)")
            self._proc.syscall()

            # must actually wait for syscall, not any event
            syscall_found = False
            ignore_callback = lambda x: False
            def syscall_callback(process, syscall):
                nonlocal syscall_found
                syscall_found = True
            break_callback = lambda: syscall_found

            # backup the old ignore_callbacks
            for process in self._debugger:
                process.syscall_state._ignore_callback = process.syscall_state.ignore_callback
            self._monitor_syscalls(None, ignore_callback, break_callback, syscall_callback, break_on_entry=True)
            # restore the old ignore_callbacks
            for process in self._debugger:
                process.syscall_state.ignore_callback = process.syscall_state._ignore_callback
                del process.syscall_state._ignore_callback
            self._proc_trapped = False

    def close(self, terminate=False):
        if self.current_target:
            if terminate:
                try:
                   self.current_target.terminate()
                except PtraceError:
                    warning("Attempted to terminate non-existent process")
                    self._debugger.deleteProcess(self.current_target)
            # when we kill the current_target, we wake up the forkserver from the trap
            self._wakeup_forkserver()

    @property
    @abstractmethod
    def current_target(self):
        pass

    def __del__(self):
        self.close()
        self._debugger.quit()
