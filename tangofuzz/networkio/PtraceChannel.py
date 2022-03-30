from . import debug, warning, info, critical

from typing import Callable
from networkio import NetworkChannel
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
from ptrace import PtraceError
from ptrace.func_call import FunctionCallOptions
from ptrace.syscall   import PtraceSyscall, SOCKET_SYSCALL_NAMES
from ptrace.tools import signal_to_exitcode
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread
from subprocess import Popen
import signal
import traceback

SOCKET_SYSCALL_NAMES = SOCKET_SYSCALL_NAMES.union(('read', 'write'))

class PtraceChannel(NetworkChannel):
    def __init__(self, pobj: Popen, **kwargs):
        super().__init__(**kwargs)
        self._pobj = pobj

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
        self._proc = self._debugger.addProcess(self._pobj.pid, is_attached=True)
        self._syscall_signum = signal.SIGTRAP
        if self._debugger.use_sysgood:
            self._syscall_signum |= 0x80

        # FIXME this is never really used; it's just a placeholder that went
        # obsolete
        default_ignore = lambda syscall: syscall.name not in SOCKET_SYSCALL_NAMES
        self.prepare_process(self._proc, default_ignore, syscall=True)

        self._monitor_executor = ThreadPoolExecutor()

    def prepare_process(self, process, ignore_callback, syscall=True):
        if not process in self._debugger:
            self._debugger.addProcess(process, is_attached=True)

        process.syscall_state.ignore_callback = ignore_callback
        if syscall:
            process.syscall()

    def process_syscall(self, process, syscall, syscall_callback, break_on_entry, **kwargs) -> bool:
        # ensure that the syscall has finished successfully before callback
        if syscall and syscall.result != -1 and \
                (break_on_entry or syscall.result is not None):
            syscall_callback(process, syscall, **kwargs)
            # calling syscall.format() takes a lot of time and should be
            # avoided in production, even if logging is disabled
            debug(f"syscall processed: [{process.pid}] {syscall.name}")
            return True
        else:
            return False

    def process_exit(self, event):
        warning(f"Process with {event.process.pid=} exited, deleting from debugger")
        warning(f"Reason: {event}")
        self._debugger.deleteProcess(event.process)
        if event.exitcode == 0:
            raise ProcessTerminatedException(f"Process with {event.process.pid=} exited normally", exitcode=0)
        elif event.signum is not None:
            raise ProcessCrashedException(f"Process with {event.process.pid=} crashed with {event.signum=}", signum=event.signum)
        elif event.exitcode == 1:
            # exitcode == 1 is usually ASan's return code when a violation is reported
            # FIXME can we read ASan's exit code from env options?
            raise ProcessCrashedException(f"Process with {event.process.pid=} crashed with {event.exitcode=}", exitcode=1)
        else:
            raise ProcessCrashedException(f"Process with {event.process.pid=} terminated abnormally with {event.exitcode=}", exitcode=event.exitcode)

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

    def process_new(self, event, ignore_callback):
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
            self.process_exec()

    def is_event_syscall(self, event):
        return isinstance(event, ProcessSignal) and event.signum == self._syscall_signum

    def process_event(self, event, ignore_callback, syscall_callback, break_on_entry, **kwargs):
        if event is None:
            return

        is_syscall = self.is_event_syscall(event)
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
            processed = self.process_syscall(event.process, syscall, syscall_callback, break_on_entry, **kwargs)

            if not processed or not break_on_entry:
                # resume the suspended process until it encounters the next syscall
                event.process.syscall()
            else:
                # the caller is responsible for resuming the target process
                pass
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
                       stop_event: Event,
                       ignore_callback: Callable[[PtraceSyscall], bool],
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
                sc = self.process_event(event, ignore_callback, syscall_callback,
                    break_on_entry, **kwargs)
                if sc is None:
                    continue
                last_process = event.process
            except ProcessEvent as e:
                self.process_event(e, ignore_callback, syscall_callback,
                    break_on_entry, **kwargs)
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
        for process in self._debugger:
            # update the ignore_callback of processes in the debugger
            process.syscall_state.ignore_callback = ignore_callback

        ## Execute monitor target
        if monitor_target:
            future = self._monitor_executor.submit(monitor_target)

        ## Listen for and process syscalls
        stop_event = Event()
        if timeout is not None:
            timeout_timer = Thread(target=self.timeout_handler, args=(stop_event,))
            timeout_timer.daemon = True
            timeout_timer.start()

        last_process = self._monitor_syscalls_internal_loop(stop_event,
                ignore_callback, break_callback, syscall_callback,
                break_on_entry, **kwargs)

        ## Return the target's result
        result = None
        if monitor_target:
            result = future.result()
        return (last_process, result)

    def terminator(self, process):
        try:
            # WARN it seems necessary to wait for the child to exit, otherwise
            # the forkserver may misbehave, and the fuzzer will receive a lot of
            # ForkChildKilledEvents
            process.terminate()
        except PtraceError as ex:
            critical(f"Attempted to terminate non-existent process ({ex})")
        finally:
            if process in self._debugger:
                self._debugger.deleteProcess(process)
        for p in process.children:
            self.terminator(p)

    def close(self, terminate, **kwargs):
        if terminate:
            self.terminator(self._proc)

    def __del__(self):
        self._monitor_executor.shutdown(wait=True)
        self.close(terminate=True)
        self._debugger.quit()
