from . import debug

from typing import Callable
from networkio import ChannelBase
from ptrace.debugger import   (PtraceDebugger,
                               PtraceProcess,
                               ProcessEvent,
                               ProcessExit,
                               ProcessSignal,
                               NewProcessEvent,
                               ProcessExecution)
from ptrace.func_call import FunctionCallOptions
from ptrace.syscall   import PtraceSyscall, SOCKET_SYSCALL_NAMES
from ptrace.tools import signal_to_exitcode
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from subprocess import Popen
import signal

SOCKET_SYSCALL_NAMES = SOCKET_SYSCALL_NAMES.union(('read', 'write'))

class PtraceChannel(ChannelBase):
    def __init__(self, pobj: Popen, tx_callback: Callable, rx_callback: Callable,
            timescale: float):
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
        self._proc.syscall_state.ignore_callback = self._ignore_callback

    def _ignore_callback(self, syscall):
        return syscall.name not in self._syscall_whitelist

    def _monitor_syscalls(self,
                       monitor_target: Callable,
                       ignore_callback: Callable[[PtraceSyscall], bool],
                       break_callback: Callable[..., bool],
                       syscall_callback: Callable[[PtraceProcess, PtraceSyscall], None],
                       break_on_entry: bool = False,
                       resume_process: bool = True,
                       **kwargs):
        def prepare_process(process, ignore_callback, syscall=True):
            if not process in self._debugger:
                debugger.addProcess(process, is_attached=True)
            process.syscall_state.ignore_callback = ignore_callback
            if syscall:
                process.syscall()

        def process_syscall(process, syscall):
            # ensure that the syscall has finished successfully before callback
            if syscall and syscall.result != -1 and \
                    (break_on_entry or syscall.result is not None):
                syscall_callback(process, syscall, **kwargs)
                debug(f"syscall processed: {syscall.format()}")
            else:
                debug(f"syscall request ignored")

        def process_event(event, reset_syscall=False, intercept_syscall=True):
            if event is None:
                return
            syscall_signum = signal.SIGTRAP
            if self._debugger.use_sysgood:
                syscall_signum |= 0x80
            is_syscall = isinstance(event, ProcessSignal) and event.signum == syscall_signum

            if not is_syscall:
                try:
                    raise event
                except ProcessExit as event:
                    exitcode = -256
                    if event.exitcode is not None:
                        exitcode = event.exitcode
                    raise ChannelBrokenException(f"Process with {event.process.pid=} exited with code {exitcode}")
                except ProcessSignal as event:
                    debug(f"Target process with {event.process.pid=} received signal with {event.signum=}")
                    event.display()
                    signum = event.signum
                    if signum == signal.SIGSTOP:
                        signum = 0
                    if reset_syscall:
                        event.process.syscall(signum)
                    else:
                        event.process.cont(signum)
                    exitcode = signal_to_exitcode(event.signum)
                    return
                except NewProcessEvent as event:
                    # monitor child for syscalls as well. may be needed for multi-thread or multi-process targets
                    debug(f"Target process forked, adding child process with {event.process.pid=} to debugger")
                    prepare_process(event.process, ignore_callback, reset_syscall)
                    if reset_syscall:
                        event.process.parent.syscall()
                    else:
                        event.process.cont()
                        event.process.parent.cont()
                    return
                except ProcessExecution as event:
                    debug(f"Target process with {event.process.pid=} called exec; removing from debugger")
                    self._debugger.deleteProcess(event.process)
                    return
            else:
                # Process syscall enter or exit
                debug(f"Target process with {event.process.pid=} requested a syscall")
                state = event.process.syscall_state

                sc = PtraceSyscall(event.process, self._syscall_options, event.process.getregs())
                debug(f"Traced {sc.name=} with {state.name=} and {state.next_event=}")

                syscall = state.event(self._syscall_options)

                if intercept_syscall:
                    process_syscall(event.process, syscall)

                if reset_syscall:
                    event.process.syscall()
                else:
                    event.process.cont()

                return syscall

        def stop_process(process):
            if not process.is_stopped and not process.isTraced():
                process.kill(signal.SIGSTOP)
                # sending a signal would restart a syscall, so we clear state
                process.syscall_state.clear()
                debug(f"Sent SIGSTOP to interrupt target with {process.pid=}")
                sigstopped = False
                while not sigstopped:
                    try:
                        # discard received SIGSTOP signal, queue up any other event
                        process.waitSignals(signal.SIGSTOP)
                        sigstopped = True
                    except ProcessEvent as event:
                        syscall = process_event(event, reset_syscall=True, intercept_syscall=False)
                        # if we receive a syscall, we add it to the queue
                        if syscall:
                            self._pending.append((event.process, syscall))

        with ThreadPoolExecutor() as executor:
            for process in self._debugger:
                prepare_process(process, ignore_callback, syscall=False)
                stop_process(process)
                process.syscall()

            ## Execute monitor target
            if monitor_target:
                future = executor.submit(monitor_target)

            ## Listen for and process syscalls
            shutdown = False
            while True:
                if not self._debugger:
                    raise ChannelBrokenException("Process was terminated while waiting for syscalls")

                if self._pending:
                    process, syscall = self._pending.pop(0)
                    process_syscall(process, syscall)
                    continue

                try:
                    event = self._debugger.waitSyscall(blocking=not shutdown)
                    if event is not None:
                        raise event
                except ProcessEvent as e:
                    event = process_event(e, reset_syscall=True, intercept_syscall=not shutdown)
                    if event is None:
                        continue
                    elif shutdown:
                        self._pending.append((event.process, event))

                if event is None:
                    break

                if not shutdown:
                    shutdown = break_callback()
                    if shutdown:
                        debug("Clearing remaining events and breaking out of debug loop")

            ## Resume processes and remove syscall breakpoints
            if resume_process:
                for process in self._debugger:
                    stop_process(process)
                    process.cont()
                    debug(f"Resumed process with {process.pid=}")

            ## Return the target's result
            if monitor_target:
                return future.result()

    def close(self):
        # if self._debugger and self._proc.is_attached:
        #     self._proc.terminate(wait_exit=True)
        self._debugger.quit()

    def __del__(self):
        self.close()