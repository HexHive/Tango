from __future__ import annotations

from tango.ptrace import debug, info, warning, critical, error
from tango.ptrace import PtraceError
from tango.ptrace.debugger import (PtraceProcess, ProcessSignal, ProcessExit,
                            ForkChildKilledEvent)
from tango.ptrace.binding import HAS_PTRACE_EVENTS, HAS_SIGNALFD
from tango.ptrace.os_tools import HAS_PROC
if HAS_PTRACE_EVENTS:
    from tango.ptrace.binding.func import (
        PTRACE_O_TRACEFORK, PTRACE_O_TRACEVFORK,
        PTRACE_O_TRACEEXEC, PTRACE_O_TRACESYSGOOD,
        PTRACE_O_TRACECLONE, PTRACE_O_TRACESECCOMP, THREAD_TRACE_FLAGS)
from tango.ptrace.binding import ptrace_detach
if HAS_PROC:
    from tango.ptrace.linux_proc import readProcessStat, ProcError
if HAS_SIGNALFD:
    from tango.ptrace.binding import (
        create_signalfd, read_signalfd, SFD_CLOEXEC, SFD_NONBLOCK)

from typing import Optional, Iterator, Sequence
from os import waitpid, close, WNOHANG, WUNTRACED
from signal import (
    SIGTRAP, SIGSTOP, SIGCHLD, SIGKILL, pthread_sigmask, SIG_BLOCK)
from errno import ECHILD
from time import sleep
from collections import deque
from lru import LRU
from contextlib import contextmanager
from dataclasses import dataclass, InitVar, replace
import asyncio

class DebuggerError(PtraceError):
    pass

WaitPidEvent = tuple[int, int, int]  # (eid, pid, status)

@dataclass(unsafe_hash=True)
class PtraceSubscription:
    wanted_pid: Optional[int]
    debugger: PtraceDebugger

    next_event: InitVar[int] = 0

    def __post_init__(self, next_event):
        self._ready = asyncio.Event()
        self.next_event = next_event

    def subscribed(self, pid):
        return self.catch_all or self.wanted_pid == pid

    def notify(self):
        self._ready.set()

    def fork(self, wanted_pid: Optional[int]=None):
        changes = {'next_event': self.next_event}
        if wanted_pid:
            changes['wanted_pid'] = wanted_pid
        forked = replace(self, **changes)
        forked.debugger.add_subscription(forked)
        return forked

    async def get(self) -> WaitPidEvent:
        while True:
            try:
                item = self.get_nowait()
                break
            except StopIteration:
                self._ready.clear()
                await self._ready.wait()
                continue
        return item

    def get_nowait(self) -> WaitPidEvent:
        idx, item = next(self.events)
        self.next_event = item[0] + 1
        return item

    @property
    def events(self) -> Iterator[tuple[int, WaitPidEvent]]:
        indexed = enumerate(self.debugger.event_history)
        by_eid = filter(lambda e: e[1][0] >= self.next_event, indexed)
        by_pid = filter(lambda e: self.subscribed(e[1][1]), by_eid)
        return by_pid

    @property
    def catch_all(self) -> bool:
        return not self.wanted_pid

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

class PtraceDebugger(object):
    """
    Debugger managing one or multiple processes at the same time.

    Methods
    =======

     * Process list:
       - addProcess(): add a new process
       - deleteProcess(): remove a process from the debugger

     * Wait for an event:
       - waitProcessEvent(): wait for a process event
       - waitSignals(): wait for a signal
       - waitSyscall(): wait for the next syscall event

     * Options:
      - traceFork(): enable fork() tracing
      - traceExec(): enable exec() tracing
      - traceClone(): enable clone() tracing
      - enableSysgood(): enable sysgood option

     * Other:
       - quit(): quit the debugger, terminate all processes

    Operations
    ==========

     - iterate on all processes: "for process in debugger: ..."
     - get a process by its identifier: "process = debugger[pid]"
     - get the number of processes: len(debugger)

    Attributes
    ==========

     - dict: processes dictionary (pid -> PtraceProcess)
     - list: processes list
     - options: ptrace options
     - trace_fork (bool): fork() tracing is enabled?
     - trace_exec (bool): exec() tracing is enabled?
     - trace_clone (bool): clone() tracing is enabled?
     - trace_seccomp (bool): seccomp_trace tracing is enabled?
     - use_sysgood (bool): sysgood option is enabled?
    """

    def __init__(self, *, loop=None):
        self.dict = {}   # pid -> PtraceProcess object
        self.list = []
        self.options = 0
        self.trace_fork = False
        self.trace_exec = False
        self.trace_clone = False
        self.trace_seccomp = False
        self.use_sysgood = False
        self._loop = loop or asyncio.get_running_loop()

        # WARN sometimes, when a traced process forks, the SIGSTOP from the
        # child arrives before the PTRACE_EVENT_CLONE, and this brings the
        # ptrace channel out of sync and it gets stuck. So we enqueue events for
        # unknown PIDs. However, we don't enqueue them forever, because the
        # tracer may also receive signals for its own child processes (see
        # WebDataLoader, create_svg launches a process). If we enqueue these
        # signals and deliver them later to traced processes, there will be
        # undefined behavior.
        self.event_history = deque()
        self._event_counter = 0
        self.subscribers = {}
        self.enableSysgood()

        if HAS_SIGNALFD:
            # FIXME should block SIGCHLD? this may interfere with the asyncio
            # child process monitor
            pthread_sigmask(SIG_BLOCK, {SIGCHLD})
            self._sigfd = create_signalfd(SIGCHLD,
                flags=SFD_CLOEXEC | SFD_NONBLOCK)

            self._sigfd_event = asyncio.Event()
            # we'll use this as a placeholder for siginfo instead of reallocing
            self._sigfd_siginfo = None
            self._loop.add_reader(self._sigfd, self._signalfd_ready_cb)
            # we flush waitpid once initially to fetch waiting signals
            self._flush_and_publish()

    if HAS_SIGNALFD:
        def _signalfd_ready_cb(self):
            try:
                self._sigfd_siginfo = read_signalfd(
                    self._sigfd, self._sigfd_siginfo)
                assert self._sigfd_siginfo.ssi_signo == SIGCHLD
                self._flush_and_publish()
            except BlockingIOError as ex:
                # this first occured when the signalfd reader was not removed
                # from the running loop
                error("signalfd reader called without queued signals.")
                raise

        def _flush_and_publish(self):
            ready = False
            while True:
                pid, status = self._waitpid_and_publish(None, blocking=False)
                if not pid:
                    break
                ready = True
            if ready:
                self._sigfd_event.set()

    def traceProcess(self, process):
        if process.pid in self.dict:
            raise KeyError(f"The pid {process.pid} is already registered!")
        self.dict[process.pid] = process
        self.list.append(process)

    async def addProcess(self, pid, is_attached, **kwargs):
        """
        Add a new process using its identifier. Use is_attached=False to
        attach an existing (running) process, and is_attached=True to trace
        a new (stopped) process.
        """
        process = PtraceProcess(self, pid, is_attached, **kwargs)
        debug("Attach %s to debugger" % process)
        self.traceProcess(process)
        try:
            await process.waitSignals(SIGTRAP, SIGSTOP)
        except KeyboardInterrupt:
            error(
                "User interrupt! Force the process %s attach "
                "(don't wait for signals)."
                % pid)
        except ProcessSignal as event:
            event.display()
        except ProcessExit as event:
            # the process was killed on creation (OOM?)
            # we just detach it and proceed as usual
            error(f"Process PID {process.pid} died on creation! Reason: {event}")
            # This is probably not needed anymore after the fix to waitExit,
            # which was likely leaving SIGKILL signals in the process struct for
            # the same PID. <- Lol, no? Probably just a pending signal sent
            # again by us when the PID was reused
            raise ForkChildKilledEvent(event.process)
        except Exception:   # noqa: E722
            process.is_attached = False
            process.detach()
            raise
        if HAS_PTRACE_EVENTS and self.options:
            process.setoptions(self.options)
        return process

    async def quit(self):
        """
        Quit the debugger: terminate all processes in reverse order.
        """
        if HAS_SIGNALFD and self._sigfd:
            self._loop.remove_reader(self._sigfd)
            close(self._sigfd)
            self._sigfd = None

        if not self.list:
            return
        debug("Quit debugger")
        # Terminate processes in reverse order
        # to kill children before parents
        processes = list(self.list)
        for process in reversed(processes):
            await process.terminate()
            process.detach()

    def kill_all(self):
        processes = list(self.list)
        for process in reversed(processes):
            process.kill(SIGKILL)

    def _waitpid(self, wanted_pid, *, blocking):
        """
        Wait for a process event from a specific process (if wanted_pid is
        set) or any process (wanted_pid=None). The call is blocking is
        blocking option is True. Return the tuple (pid, status).

        See os.waitpid() documentation for explanations about the result.
        """
        flags = 0 | WUNTRACED
        if not blocking:
            flags |= WNOHANG
        if wanted_pid:
            if wanted_pid not in self.dict:
                raise DebuggerError("Unknown PID: %r" %
                                    wanted_pid, pid=wanted_pid)

            process = self.dict[wanted_pid]
            for p in process.children:
                if p.is_thread:
                    flags |= THREAD_TRACE_FLAGS
                    break

        try:
            if wanted_pid and (flags & THREAD_TRACE_FLAGS) == 0:
                pid, status = waitpid(wanted_pid, flags)
            else:
                pid, status = waitpid(-1, flags)
        except ChildProcessError:
            pid = status = None
        else:
            if (blocking or pid) and wanted_pid and (pid != wanted_pid) and \
                    (pid not in map(lambda p: p.pid, filter(lambda p: p.is_thread,
                        process.children))):
                raise DebuggerError("Unwanted PID: %r (instead of %s)"
                                % (pid, wanted_pid), pid=pid)
        return pid, status

    def _waitpid_and_publish(self, *args, **kwargs):
        result = self._waitpid(*args, **kwargs)
        if result[0]:
            pid, status = result
            pidevent = self._publish_status(pid, status)
        return result

    def _publish_status(self, pid, status) -> WaitPidEvent:
        assert pid
        eid = self._event_counter
        self._event_counter += 1

        published = False
        for sub in (pid, None):  # we check both specific and catch-all subs
            if subs := self.subscribers.get(sub):
                for subscription in subs:
                    subscription.notify()
                    published = True
                # if a more specific subscription (than catch-all) is available,
                # we only publish to that subscription
                break

        # otherwise, we enqueue it and hope someone picks it up in time
        if not published:
            warning(f"No subscribers for {pid=}")

        pidevent = (eid, pid, status)
        self.event_history.append(pidevent)
        return pidevent

    def add_subscription(self, subscription: PtraceSubscription):
        if not (others := self.subscribers.get(subscription.wanted_pid)):
            self.subscribers[subscription.wanted_pid] = {subscription}
        else:
            others.add(subscription)

    def subscribe(self, wanted_pid: Optional[int]=None, /,
            *, start_at=0, **kwargs) -> PtraceSubscription:
        subscription = PtraceSubscription(wanted_pid, self, next_event=start_at)
        self.add_subscription(subscription)
        return subscription

    def unsubscribe(self, subscription: PtraceSubscription):
        # others may have subscribed in the meantime, so we remove ourselves
        # and check again
        others = self.subscribers[subscription.wanted_pid]
        others.discard(subscription)
        if not others:
            self.subscribers.pop(subscription.wanted_pid)

    @contextmanager
    def subscription(self, wanted_pid, subscription=None, **kwargs):
        should_unsub = not subscription
        if should_unsub:
            subscription = self.subscribe(wanted_pid, **kwargs)
        else:
            assert not subscription.wanted_pid or \
                subscription.wanted_pid == wanted_pid

        try:
            yield subscription
        finally:
            if should_unsub:
                self.unsubscribe(subscription)

    async def _wait_status(self, wanted_pid, subscription,
            *, blocking) -> WaitPidEvent:
        while True:
            match (HAS_SIGNALFD, blocking, subscription.ready):
                case (True, True, _) | (_, _, True):
                    item = await subscription.get()
                case (True, False, False):
                    # yield to the event loop, let it poll signalfd
                    await asyncio.sleep(0)
                    try:
                        item = subscription.get_nowait()
                    except StopIteration:
                        item = (None, None)
                case (False, _, False):
                    _ = self._waitpid_and_publish(wanted_pid, blocking=blocking)
                    # we fetch it from the queue as well to keep it in sync
                    item = await subscription.get()
                case _:
                    raise ValueError("Unexpected match parameters")

            eid, pid, status = item
            if wanted_pid and wanted_pid != pid:
                # subscription is catch-all but wait_status was called with a
                # specific pid
                debug(f"Re-ordering event {item}")
                # FIXME maybe use del arr[idx], must get idx from subscription
                self.event_history.remove(item)
                if HAS_SIGNALFD:
                    # yield control to the event loop to populate sigqueue
                    await asyncio.sleep(0)
                self._publish_status(pid, status)
                # the event is now at the tail of the event history
                continue
            break
        return item

    async def _wait_event(self, wanted_pid, *, blocking=True, subscription=None):
        """
        Wait for a process event from the specified process identifier. If
        blocking=False, return None if there is no new event, otherwise return
        an object based on ProcessEvent.
        """
        process = None
        with self.subscription(wanted_pid, subscription) as subscription:
            while not process:
                republish = False
                try:
                    eid, pid, status = await self._wait_status(
                        wanted_pid, subscription, blocking=blocking)
                    debug(f"wait_event({wanted_pid}): "
                          f"subscriber to {subscription.wanted_pid} "
                          f"received event {eid}")
                except OSError as err:
                    if wanted_pid and err.errno == ECHILD:
                        process = self.dict[wanted_pid]
                        return await process.processTerminated()
                    else:
                        raise err
                if not blocking and not pid:
                    return None
                try:
                    process = self.dict[pid]
                except KeyError:
                    if HAS_PROC:
                        try:
                            stat = readProcessStat(pid)
                            if parent := self.dict.get(stat.ppid):
                                republish = True
                                debug(f"Received premature signal for"
                                      f" a child ({pid=}) of"
                                      f" a traced process ({parent=})")
                            else:
                                debug(f"Ignoring signal for unknown {pid=}")
                        except ProcError:
                            warning(f"Process ({pid=}) died before"
                                     " its signal could be processed")
                    else:
                        republish = True
                        debug(f"Received signal for unknown {pid=},"
                               " placing event back in queue")

                if republish:
                    # FIXME this may result in duplicate events for some subs?
                    reordered = self._publish_status(pid, status)
                    debug(f"Republished {eid} as {reordered[0]}")
                    if HAS_SIGNALFD:
                        # yield control to the event loop to populate
                        # sigqueue
                        await asyncio.sleep(0)

        # flush early event history if too long
        if (l := len(self.event_history)) & ~(0x100 - 1):
            min_eid = min(sub.next_event
                for _, subs in self.subscribers.items()
                    for sub in subs)
            while self.event_history and self.event_history[0][0] < min_eid:
                del self.event_history[0]
            debug(f"Event history too long (len={l});"
                    f" purged {l - len(self.event_history)} items.")

        return await process.processStatus(status)

    async def waitProcessEvent(self, pid=None, **kwargs):
        """
        Wait for a process event from a specific process (if pid option is
        set) or any process (default). If blocking=False, return None if there
        is no new event, otherwise return an object based on ProcessEvent.
        """
        return await self._wait_event(pid, **kwargs)

    async def waitSignals(self, *signals, pid=None, **kwargs):
        """
        Wait for any signal or some specific signals (if specified) from a
        specific process (if pid keyword is set) or any process (default).
        Return a ProcessSignal object or raise an unexpected ProcessEvent.
        """
        event = await self._wait_event(pid, **kwargs)
        if event is None:
            return
        if event.__class__ != ProcessSignal:
            raise event
        signum = event.signum
        if signum in signals or not signals:
            return event
        raise event

    async def waitSyscall(self, process=None, **kwargs):
        """
        Wait for the next syscall event (enter or exit) for a specific process
        (if specified) or any process (default). Return a ProcessSignal object
        or raise an unexpected ProcessEvent.
        """
        if process:
            event = await self.waitProcessEvent(pid=process.pid, **kwargs)
        else:
            event = await self.waitProcessEvent(**kwargs)
        if event is None:
            return
        if event.is_syscall_stop():
            return event
        else:
            raise event

    def deleteProcess(self, process=None, pid=None):
        """
        Delete a process from the process list.
        """
        if not process:
            try:
                process = self.dict[pid]
            except KeyError:
                return
        self.dict.pop(process.pid, None)
        try:
            self.list.remove(process)
        except ValueError:
            return
        debug(f"Deleted {process} from debugger")

    def updateProcessOptions(self):
        for process in self:
            process.setoptions(self.options)

    def traceFork(self):
        """
        Enable fork() tracing. Do nothing if it's not supported.
        """
        if not HAS_PTRACE_EVENTS:
            raise DebuggerError(
                "Tracing fork events is not supported on this architecture or operating system")
        self.options |= PTRACE_O_TRACEFORK | PTRACE_O_TRACEVFORK
        self.trace_fork = True
        self.updateProcessOptions()
        debug("Debugger trace forks (options=%s)" % self.options)

    def traceExec(self):
        """
        Enable exec() tracing. Do nothing if it's not supported.
        """
        if not HAS_PTRACE_EVENTS:
            # no effect on OS without ptrace events
            return
        self.trace_exec = True
        self.options |= PTRACE_O_TRACEEXEC
        self.updateProcessOptions()
        debug("Debugger trace execs (options=%s)" % self.options)

    def traceClone(self):
        """
        Enable clone() tracing. Do nothing if it's not supported.
        """
        if not HAS_PTRACE_EVENTS:
            # no effect on OS without ptrace events
            return
        self.trace_clone = True
        self.options |= PTRACE_O_TRACECLONE
        debug("Debugger trace execs (options=%s)" % self.options)
        self.updateProcessOptions()

    def traceSeccomp(self):
        if not HAS_PTRACE_EVENTS:
            # no effect on OS without ptrace events
            return
        self.trace_seccomp = True
        self.options |= PTRACE_O_TRACESECCOMP
        debug("Debugger trace execs (options=%s)" % self.options)
        self.updateProcessOptions()

    def enableSysgood(self):
        """
        Enable sysgood option: ask the kernel to set bit #7 of the signal
        number if the signal comes from the kernel space. If the signal comes
        from the user space, the bit is unset.
        """
        if not HAS_PTRACE_EVENTS:
            # no effect on OS without ptrace events
            return
        self.use_sysgood = True
        self.options |= PTRACE_O_TRACESYSGOOD

    @property
    def sigtrap_signum(self):
        signum = SIGTRAP
        if self.use_sysgood:
            signum |= 0x80
        return signum

    def __getitem__(self, pid):
        return self.dict[pid]

    def __iter__(self):
        return iter(self.list)

    def __len__(self):
        return len(self.list)

    def __del__(self):
        self.kill_all()