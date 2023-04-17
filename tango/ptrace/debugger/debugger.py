from tango.ptrace import debug, info, warning, error
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
    import asyncio

from os import waitpid, WNOHANG, WUNTRACED
from signal import (
    SIGTRAP, SIGSTOP, SIGCHLD, SIGKILL, pthread_sigmask, SIG_BLOCK)
from errno import ECHILD
from time import sleep
from collections import defaultdict
from lru import LRU

class DebuggerError(PtraceError):
    pass


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

        # WARN sometimes, when a traced process forks, the SIGSTOP from the
        # child arrives before the PTRACE_EVENT_CLONE, and this brings the
        # ptrace channel out of sync and it gets stuck. So we enqueue events for
        # unknown PIDs. However, we don't enqueue them forever, because the
        # tracer may also receive signals for its own child processes (see
        # WebDataLoader, create_svg launches a process). If we enqueue these
        # signals and deliver them later to traced processes, there will be
        # undefined behavior.
        self.sig_queue = LRU(10)
        self.sig_queue.set_callback(self._sig_queue_evict)
        self.sig_evicted = list()
        self.enableSysgood()

        if HAS_SIGNALFD:
            loop = loop or asyncio.get_running_loop()
            # FIXME should block SIGCHLD? this may interfere with the asyncio
            # child process monitor
            pthread_sigmask(SIG_BLOCK, {SIGCHLD})
            self._sigfd = create_signalfd(SIGCHLD,
                flags=SFD_CLOEXEC | SFD_NONBLOCK)

            self._sigfd_event = asyncio.Event()
            # we'll use this as a placeholder for siginfo instead of reallocing
            self._sigfd_siginfo = None
            loop.add_reader(self._sigfd, self._signalfd_ready_cb)
            # we flush waitpid once initially to fetch queued signals
            self._flush_and_enqueue()

    def _sig_queue_evict(self, pid, queue, always_expand=True):
        if pid not in self.dict:
            # hard luck; we could not find an owner in time
            return

        if always_expand or all(p in self.dict for p in self.sig_queue.keys()):
            # the LRU is full, but all PIDs belong to the debugger;
            # we expand the LRU so as not to lose signals for known PIDs
            self.sig_queue.set_size(self.sig_queue.get_size() * 2)
            warning("Expanded LRU cache due to necessary eviction")

        # we keep track of known evicted PIDs to be re-used later
        self.sig_evicted.append((pid, queue))

    def traceProcess(self, process):
        if process.pid in self.dict:
            raise KeyError(f"The pid {process.pid} is already registered!")
        self.dict[process.pid] = process
        self.list.append(process)

    async def addProcess(self, pid, is_attached, parent=None, is_thread=False):
        """
        Add a new process using its identifier. Use is_attached=False to
        attach an existing (running) process, and is_attached=True to trace
        a new (stopped) process.
        """
        process = PtraceProcess(self, pid, is_attached,
                                parent=parent, is_thread=is_thread)
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

    def quit(self):
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
            process.terminate()
            process.detach()

    if HAS_SIGNALFD:
        def _signalfd_ready_cb(self):
            self._sigfd_siginfo = read_signalfd(self._sigfd, self._sigfd_siginfo)
            assert self._sigfd_siginfo.ssi_signo == SIGCHLD
            self._flush_and_enqueue()

        def _flush_and_enqueue(self):
            ready = False
            while True:
                pid, status = self._waitpid(None, blocking=False)
                if pid <= 0:
                    break
                self._push_status(pid, status)
                ready = True
            if ready:
                self._sigfd_event.set()

    def _waitpid(self, wanted_pid, blocking=True):
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

        if wanted_pid and (flags & THREAD_TRACE_FLAGS) == 0:
            pid, status = waitpid(wanted_pid, flags)
        else:
            pid, status = waitpid(-1, flags)

        if (blocking or pid) and wanted_pid and (pid != wanted_pid) and \
                (pid not in map(lambda p: p.pid, filter(lambda p: p.is_thread,
                    process.children))):
            raise DebuggerError("Unwanted PID: %r (instead of %s)"
                                % (pid, wanted_pid), pid=pid)
        return pid, status

    def _dequeue(self, pid=None):
        if pid:
            if queue := self.sig_queue.get(pid):
                return pid, queue
            for p, queue in self.sig_evicted:
                if p == pid:
                    return pid, queue
        else:
            if self.sig_evicted:
                rv = self.sig_evicted.pop(0)
            else:
                rv = self.sig_queue.peek_last_item()
            return rv
        return None

    async def _fetch_status(self, wanted_pid, blocking=True):
        queue = None
        if HAS_SIGNALFD and blocking:
            while True:
                rv = self._dequeue(wanted_pid)
                if rv:
                    wanted_pid, queue = rv
                    break
                await self._sigfd_event.wait()
                self._sigfd_event.clear()
            # at this point, we must only fetch from a queue
            assert queue

        if not queue:
            if not (rv := self._dequeue(wanted_pid)):
                return self._waitpid(wanted_pid, blocking=blocking)
            else:
                debug(f"waitpid(): Fetching signal for PID {wanted_pid} from queue")
                wanted_pid, queue = rv

        status = queue.pop(0)
        if queue and not wanted_pid in self.sig_queue:
            # must have been popped from the eviction list
            self.sig_queue[wanted_pid] = queue
        elif not queue:
            self.sig_queue.pop(wanted_pid, None)
        return wanted_pid, status

    def _push_status(self, pid, status):
        if not (queue := self.sig_queue.get(pid)):
            queue = self.sig_queue[pid] = list()
        queue.append(status)

    async def _wait_event(self, wanted_pid, blocking=True):
        """
        Wait for a process event from the specified process identifier. If
        blocking=False, return None if there is no new event, otherwise return
        an object based on ProcessEvent.
        """
        process = None
        while not process:
            try:
                pid, status = await self._fetch_status(wanted_pid)
            except OSError as err:
                if wanted_pid and err.errno == ECHILD:
                    process = self.dict[wanted_pid]
                    return process.processTerminated()
                else:
                    raise err
            if not blocking and not pid:
                return None
            try:
                process = self.dict[pid]
            except KeyError:
                enq = True
                if HAS_PROC:
                    try:
                        stat = readProcessStat(pid)
                        if parent := self.dict.get(stat.ppid):
                            debug(f"Received premature signal for"
                                  f" a child ({pid=}) of"
                                  f" a traced process ({parent=})")
                        else:
                            enq = False
                            debug(f"Ignoring signal for unknown {pid=}")
                    except ProcError:
                        enq = False
                        warning(f"Process ({pid=}) died before"
                                 " its signal could be processed")
                else:
                    debug(f"Received signal for unknown {pid=},"
                           " placing event in queue")
                if enq:
                    self._push_status(pid, status)
                    if HAS_SIGNALFD:
                        # yield control to the event loop to populate sigqueue
                        await asyncio.sleep(0)
                continue

        return await process.processStatus(status)

    async def waitProcessEvent(self, pid=None, blocking=True):
        """
        Wait for a process event from a specific process (if pid option is
        set) or any process (default). If blocking=False, return None if there
        is no new event, otherwise return an object based on ProcessEvent.
        """
        return await self._wait_event(pid, blocking=blocking)

    async def waitSignals(self, *signals, pid=None, blocking=True):
        """
        Wait for any signal or some specific signals (if specified) from a
        specific process (if pid keyword is set) or any process (default).
        Return a ProcessSignal object or raise an unexpected ProcessEvent.
        """
        event = await self._wait_event(pid, blocking=blocking)
        if event is None:
            return
        if event.__class__ != ProcessSignal:
            raise event
        signum = event.signum
        if signum in signals or not signals:
            return event
        raise event

    async def waitSyscall(self, process=None, blocking=True):
        """
        Wait for the next syscall event (enter or exit) for a specific process
        (if specified) or any process (default). Return a ProcessSignal object
        or raise an unexpected ProcessEvent.
        """
        if process:
            event = await self.waitProcessEvent(pid=process.pid, blocking=blocking)
        else:
            event = await self.waitProcessEvent(blocking=blocking)
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
            pass

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
