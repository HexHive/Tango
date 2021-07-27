from . import debug

from typing import Callable
from networkio import ChannelBase
from   common      import (ChannelBrokenException,
                          ChannelSetupException)
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
from subprocess import Popen
import signal

SOCKET_SYSCALL_NAMES = SOCKET_SYSCALL_NAMES.union(('read', 'write'))

class PtraceChannel(ChannelBase):
    def __init__(self, pobj: Popen, tx_callback: Callable, rx_callback: Callable,
            timescale: float):
        super().__init__(pobj, tx_callback, rx_callback, timescale)

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
        self._prepare_process(self._proc, self._ignore_callback, syscall=True)

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
                       **kwargs):

        def process_syscall(process, syscall):
            # ensure that the syscall has finished successfully before callback
            if syscall and syscall.result != -1 and \
                    (break_on_entry or syscall.result is not None):
                syscall_callback(process, syscall, **kwargs)
                debug(f"syscall processed: {syscall.format()}")
            else:
                # debug(f"syscall request ignored")
                pass

        def process_auxiliary_event(event):
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
                event.process.syscall(signum)
                exitcode = signal_to_exitcode(event.signum)
                return
            except NewProcessEvent as event:
                # monitor child for syscalls as well. may be needed for multi-thread or multi-process targets
                debug(f"Target process forked, adding child process with {event.process.pid=} to debugger")
                self._prepare_process(event.process, ignore_callback, syscall=True)
                event.process.parent.syscall()
                return
            except ProcessExecution as event:
                debug(f"Target process with {event.process.pid=} called exec; removing from debugger")
                self._debugger.deleteProcess(event.process)
                return

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

                ### DEBUG ###
                # sc = PtraceSyscall(event.process, self._syscall_options, event.process.getregs())
                # debug(f"Traced {sc.name=} with {state.name=} and {state.next_event=}")
                #############

                syscall = state.event(self._syscall_options)

                process_syscall(event.process, syscall)

                event.process.syscall()
                return syscall

        with ThreadPoolExecutor() as executor:
            for process in self._debugger:
                # update the ignore_callback of processes in the debugger
                self._prepare_process(process, ignore_callback, syscall=False)

            ## Execute monitor target
            if monitor_target:
                future = executor.submit(monitor_target)

            ## Listen for and process syscalls
            while True:
                if not self._debugger:
                    raise ChannelBrokenException("Process was terminated while waiting for syscalls")

                try:
                    event = self._debugger.waitSyscall()
                    # even if it is a syscall, we raise it to pass it to process_event()
                    raise event
                except ProcessEvent as e:
                    syscall = process_event(e)
                    if syscall is None:
                        continue

                if break_callback():
                    debug("Syscall monitoring finished, breaking out of debug loop")
                    break

            ## Return the target's result
            if monitor_target:
                return future.result()

    def close(self):
        for process in self._debugger:
            try:
                process.cont()
            except:
                pass
        try:
            self._debugger.quit()
        except:
            pass

    def __del__(self):
        self.close()