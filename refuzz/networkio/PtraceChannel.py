from . import debug, warning, info

from typing import Callable
from networkio import ChannelBase
from   common      import (ChannelTimeoutException,
                          ProcessCrashedException,
                          ProcessTerminatedException)
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
from threading import Event, Thread
from subprocess import Popen
import signal
import traceback

SOCKET_SYSCALL_NAMES = SOCKET_SYSCALL_NAMES.union(('read', 'write'))

class PtraceChannel(ChannelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        self._syscall_signum = signal.SIGTRAP
        if self._debugger.use_sysgood:
            self._syscall_signum |= 0x80

        # FIXME this is never really used; it's just a placeholder that went
        # obsolete
        default_ignore = lambda syscall: syscall.name not in self._syscall_whitelist
        self.prepare_process(self._proc, default_ignore, syscall=True)

    def prepare_process(self, process, ignore_callback, syscall=True):
        if not process in self._debugger:
            self._debugger.addProcess(process, is_attached=True)

        process.syscall_state.ignore_callback = ignore_callback
        if syscall:
            process.syscall()

    def process_syscall(self, process, syscall, syscall_callback, break_on_entry, **kwargs):
        # ensure that the syscall has finished successfully before callback
        if syscall and syscall.result != -1 and \
                (break_on_entry or syscall.result is not None):
            syscall_callback(process, syscall, **kwargs)
            debug(f"syscall processed: [{process.pid}] {syscall.format()}")

    def process_exit(self, event):
        exitcode = -256
        if event.exitcode is not None:
            exitcode = event.exitcode
        warning(f"Process with {event.process.pid=} exited, deleting from debugger")
        warning(f"Reason: {event}")
        self._debugger.deleteProcess(event.process)
        if exitcode == 0:
            raise ProcessTerminatedException(f"Process with {event.process.pid=} exited normally", exitcode=0)
        elif exitcode in (1, *range(128, 128 + 65)):
            raise ProcessCrashedException(f"Process with {event.process.pid=} crashed with code {exitcode}", exitcode=exitcode)
        else:
            raise ProcessTerminatedException(f"Process with {event.process.pid=} exited with code {exitcode}", exitcode=exitcode)

    def process_signal(self, event):
        if event.signum == signal.SIGUSR2:
            raise ChannelTimeoutException("Channel timeout when waiting for syscall")
        elif event.signum in (signal.SIGINT, signal.SIGWINCH):
            debug(f"Process with {event.process.pid=} received SIGINT or SIGWINCH {event.signum=}")
            # Ctrl-C or resizing the window should not be passed to child
            event.process.syscall()
            return
        elif event.signum == signal.SIGSTOP:
            critical(f"{event.process.pid=} received rogue SIGSTOP, resuming for now")
            event.process.syscall()
            return
        debug(f"Target process with {event.process.pid=} received signal with {event.signum=}")
        event.display(log=warning)
        signum = event.signum
        event.process.syscall(signum)
        exitcode = signal_to_exitcode(event.signum)

    def process_new(self, event):
        # monitor child for syscalls as well. may be needed for multi-thread or multi-process targets
        debug(f"Target process with {event.process.parent.pid=} forked, adding child process with {event.process.pid=} to debugger")
        if event.process.is_attached:
            # sometimes, child process might have been killed at creation,
            # so the debugger detaches it; we check for that here
            self.prepare_process(event.process, ignore_callback, syscall=True)
        event.process.parent.syscall()

    def process_exec(self, event):
        debug(f"Target process with {event.process.pid=} called exec; removing from debugger")
        self._debugger.deleteProcess(event.process)

    def process_auxiliary_event(self, event):
        try:
            raise event
        except ProcessExit as event:
            self.process_exit(event)
        except ProcessSignal as event:
            self.process_signal(event)
        except NewProcessEvent as event:
            self.process_new(event)
        except ProcessExecution as event:
            self.process_exec()

    def is_event_syscall(self, event):
        return isinstance(event, ProcessSignal) and event.signum == self._syscall_signum

    def process_event(self, event, syscall_callback, break_on_entry, **kwargs):
        if event is None:
            return

        is_syscall = self.is_event_syscall(event)
        if not is_syscall:
            self.process_auxiliary_event(event)
        else:
            # Process syscall enter or exit
            # debug(f"Target process with {event.process.pid=} requested a syscall")
            state = event.process.syscall_state

            ### DEBUG ###
            # sc = PtraceSyscall(event.process, self._syscall_options, event.process.getregs())
            # debug(f"Traced {sc.name=} with {state.name=} and {state.next_event=}")
            #############

            syscall = state.event(self._syscall_options)
            self.process_syscall(event.process, syscall, syscall_callback, break_on_entry, **kwargs)

            # resume the suspended process until it encounters the next syscall
            event.process.syscall()
            return syscall

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

    def _wait_for_syscall(self):
        return self._debugger.waitSyscall()

    def _monitor_syscalls_internal_loop(self,
                       break_callback: Callable[..., bool],
                       syscall_callback: Callable[[PtraceProcess, PtraceSyscall], None],
                       break_on_entry: bool = False,
                       **kwargs):
        last_process = None
        while True:
            if not self._debugger:
                raise ProcessTerminatedException("Process was terminated while waiting for syscalls", exitcode=None)

            try:
                debug("Waiting for syscall...")
                # if waitSyscall does not raise an exception, then event is
                # a syscall, otherwise it's some other ProcessEvent
                event = self._wait_for_syscall()
                if event is None:
                    continue
                syscall = self.process_event(event, syscall_callback, break_on_entry, **kwargs)
                if syscall is None:
                    continue
                last_process = event.process
            except ProcessEvent as e:
                self.process_event(e, syscall_callback, break_on_entry, **kwargs)
                continue

            if break_callback():
                debug("Syscall monitoring finished, breaking out of debug loop")
                stop_event.set()
                break
        return last_process

    def monitor_syscalls(self,
                       monitor_target: Callable,
                       ignore_callback: Callable[[PtraceSyscall], bool],
                       break_callback: Callable[..., bool],
                       syscall_callback: Callable[[PtraceProcess, PtraceSyscall], None],
                       break_on_entry: bool = False,
                       timeout: float = None,
                       **kwargs):
        with ThreadPoolExecutor() as executor:
            for process in self._debugger:
                # update the ignore_callback of processes in the debugger
                self.prepare_process(process, ignore_callback, syscall=False)

            ## Execute monitor target
            if monitor_target:
                future = executor.submit(monitor_target)

            ## Listen for and process syscalls
            stop_event = Event()
            if timeout is not None:
                timeout_timer = Thread(target=self.timeout_handler, args=(stop_event,))
                timeout_timer.daemon = True
                timeout_timer.start()

            last_process = self._monitor_syscalls_internal_loop(
                    break_callback, syscall_callback, break_on_entry, **kwargs)

            ## Return the target's result
            result = None
            if monitor_target:
                result = future.result()
            return (last_process, result)

    def close(self):
        pass

    def __del__(self):
        self.close()
        self._debugger.quit()
