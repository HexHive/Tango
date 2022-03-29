from . import debug, warning, info, critical

from loader import StateLoaderBase
from typing import Callable
from collections.abc import Coroutine
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
from enum import Enum, auto
from dataclasses import dataclass
from asyncio import Future, Task, CancelledError, TimeoutError as AsyncTimeoutError
import signal
import traceback

class PtraceLoader(StateLoaderBase):
    def __init__(self, **kwargs):
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
        self._syscall_signum = signal.SIGTRAP
        if self._debugger.use_sysgood:
            self._syscall_signum |= 0x80

        self._ignore_callback = lambda syscall: False

        # self._proc = self._debugger.addProcess(self._pobj.pid, is_attached=True)
        # self.prepare_process(self._proc, self._ignore_callback, syscall=True)

        self._subscribers = []

    def prepare_process(self, process, ignore_callback, syscall=True):
        if not process in self._debugger:
            self._debugger.addProcess(process, is_attached=True)

        process.syscall_state.ignore_callback = ignore_callback
        if syscall:
            process.syscall()

    async def event(self, process, syscall, subscribers):
        subqs = {}
        for s in subscribers:
            ret = s.process(syscall)
            subqs[s] = []
            if s.state == SubscriberState.RUNNING \
                    and ret == EventResult.CONSUME:
                pass
            elif s.state == SubscriberState.RUNNING \
                    and ret == EventResult.DEFER:
                subqs[s].append((process, syscall))
                s.state = SubscriberState.WAITING
            elif s.state == SubscriberState.WAITING \
                    and ret == EventResult.DEFER:
                subqs[s].append((process, syscall))
            elif s.state == SubscriberState.WAITING \
                    and ret == EventResult.CONSUME:
                s.state = SubscriberState.RUNNING
                tmpq = []
                while s.state != SubscriberState.ZOMBIE and s.q:
                    p, e = s.q.pop(0)
                    subq = await self.event(p, e, [s])[s]
                    tmpq.extend(subq)
                subqs[s].extend(tmpq)
            elif ret == EventResult.BREAK:
                s.state = SubscriberState.ZOMBIE
                if s.monitor_target is None:
                    s.result.set_result((process, None))
                    try:
                        await s.timeout_task
                    except AsyncTimeoutError:
                        # the timeout will be set inside the timeout handler
                        pass
                else:
                    try:
                        res = await s.timeout_task
                    except AsyncTimeoutError:
                        # the timeout will be set inside the timeout handler
                        pass
                    else:
                        s.result.set_result((process, res))
        return subqs

    async def loop(self):
        while True:
            e = await self.wait_event()
            if self.is_event_syscall(e):
                state = e.process.syscall_state
                syscall = state.event(self._syscall_options)

                subscribers = self.get_matching_subscribers(syscall)
                subqs = await self.event(e.process, syscall, subscribers)
                for s in subscribers:
                    if s.state == SubscriberState.ZOMBIE:
                        self.unsubscribe(s)
                    else:
                        s.q.extend(subqs[s])
                        # Final passes over the queue: keep flushing the queue
                        # either until the subscriber is no longer RUNNING or
                        # the queue is empty
                        while s.state == SubscriberState.RUNNING and s.q:
                            subq = await self.event(*(s.q.pop(0)), [s])[s]
                            s.q.extend(subq)

                e.process.syscall()
            else:
                try:
                    self.process_auxiliary_event(e)
                except Exception as ex:
                    for s in self._subscribers:
                        if s.monitor_target is not None:
                            s.monitor_target.cancel(f"Exception raised: {ex}")
                        s.result.set_exception(ex)
                    s._subscribers.clear()

    async def wait_event(self):
        if not self._debugger:
            raise ProcessTerminatedException("Process was terminated while waiting for syscalls", exitcode=None)

        e = await self._debugger.waitProcessEvent()
        return e

    def process_auxiliary_event(self, event):
        if isinstance(event, NewProcessEvent):
            self.process_new(event)
        elif isinstance(event, ProcessExit):
            self.process_exit(event)
        elif isinstance(event, ProcessSignal):
            self.process_signal(event)
        elif isinstance(event, ProcessExecution):
            self.process_exec()

    def subscribe(self,
            ignore_callback: Callable[[PtraceSyscall], bool],
            process_callback: Callable[[PtraceProcess, PtraceSyscall], None],
            break_on_entry: bool = False,
            monitor_target: Coroutine = None,
            timeout: float = None):
        sub = Subscriber(ignore_callback, process_callback, break_on_entry)
        if monitor_target is not None:
            wrapped = self.timeout_handler(monitor_target, sub)
        else:
            wrapped = self.timeout_handler(sub.result, sub)
        timeout_task = asyncio.create_task(asyncio.wait_for(wrapped, timeout))

        sub.monitor_target = monitor_target
        sub.timeout_task = timeout_task
        sub.result = Future()
        sub.loader = self

        self._subscribers.append(sub)
        return sub

    def unsubscribe(self, sub):
        if sub.monitor_target is not None and not sub.monitor_target.done():
            sub.monitor_target.cancel()
        if not sub.timeout_task.done():
            sub.timeout_task.cancel()
        if not sub.result.done():
            sub.result.set_exception(CancelledError("Unsubscribed!"))
        if sub in self._subscribers:
            self._subscribers.remove(sub)

    async def timeout_handler(self, aw, sub):
        try:
            return await aw
        except CancelledError:
            # FIXME does cancellation happen in cases other than timeout?
            sub.result.set_exception(AsyncTimeoutError())
            self.unsubscribe(sub)

    def is_event_syscall(self, event):
        return isinstance(event, ProcessSignal) \
            and event.signum == self._syscall_signum

    def get_matching_subscribers(self, syscall):
        is_entry = syscall.result is None
        filter_entry = filter(lambda s: not is_entry \
            or (is_entry and s.break_on_entry), self._subscribers)
        filter_ignore = filter(lambda s: not s.ignore(syscall), filter_entry)
        return list(filter_ignore)

    def process_new(self, event, ignore_callback):
        # monitor child for syscalls as well. may be needed for multi-thread or multi-process targets
        debug(f"Target process with {event.process.parent.pid=} forked, adding child process with {event.process.pid=} to debugger")
        if event.process.is_attached:
            # sometimes, child process might have been killed at creation,
            # so the debugger detaches it; we check for that here
            self.prepare_process(event.process, ignore_callback, syscall=True)
        event.process.parent.syscall()

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
            # FIXME can we read ASan's exit code from env options or from memory?
            raise ProcessCrashedException(f"Process with {event.process.pid=} crashed with {event.exitcode=}", exitcode=1)
        else:
            raise ProcessCrashedException(f"Process with {event.process.pid=} terminated abnormally with {event.exitcode=}", exitcode=event.exitcode)

    def process_signal(self, event):
        if event.signum in (signal.SIGINT, signal.SIGWINCH):
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

    def process_exec(self, event):
        debug(f"Target process with {event.process.pid=} called exec; removing from debugger")
        self._debugger.deleteProcess(event.process)

class SubscriberState(Enum):
    RUNNING = auto()
    WAITING = auto()
    ZOMBIE  = auto()

class EventResult(Enum):
    CONSUME = auto()
    DEFER   = auto()
    BREAK   = auto()

@dataclass
class Subscriber:
    ignore         : Callable[[PtraceSyscall], bool]
    process        : Callable[[PtraceProcess, PtraceSyscall], EventResult]
    break_on_entry : bool
    monitor_target : Coroutine = None
    timeout_task   : Task = None
    result         : Future = None

    ## Internal
    state          : SubscriberState = SubscriberState.RUNNING
    loader         : PtraceLoader = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, exc_traceback):
        self.loader.unsubscribe(self)
