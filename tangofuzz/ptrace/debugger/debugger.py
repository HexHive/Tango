from .. import debug, info, warning, error
from ptrace import PtraceError
from os import waitpid, WNOHANG, WUNTRACED
from signal import SIGTRAP, SIGSTOP, SIGCHLD, SIGKILL
from errno import ECHILD
from ptrace.debugger import (PtraceProcess, ProcessSignal, ProcessExit,
                            ForkChildKilledEvent)
from ptrace.binding import HAS_PTRACE_EVENTS
from ptrace.os_tools import HAS_PROC
from time import sleep
from collections import defaultdict
if HAS_PTRACE_EVENTS:
    from ptrace.binding.func import (
        PTRACE_O_TRACEFORK, PTRACE_O_TRACEVFORK,
        PTRACE_O_TRACEEXEC, PTRACE_O_TRACESYSGOOD,
        PTRACE_O_TRACECLONE, THREAD_TRACE_FLAGS)
from ptrace.binding import ptrace_detach
from lru import LRU
if HAS_PROC:
    from ptrace.linux_proc import readProcessStat, ProcError

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
     - use_sysgood (bool): sysgood option is enabled?
    """

    def __init__(self):
        self.dict = {}   # pid -> PtraceProcess object
        self.list = []
        self.options = 0
        self.trace_fork = False
        self.trace_exec = False
        self.trace_clone = False
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
        self.enableSysgood()

    def addProcess(self, pid, is_attached, parent=None, is_thread=False):
        """
        Add a new process using its identifier. Use is_attached=False to
        attach an existing (running) process, and is_attached=True to trace
        a new (stopped) process.
        """
        if pid in self.dict:
            raise KeyError("The process %s is already registered!" % pid)
        process = PtraceProcess(self, pid, is_attached,
                                parent=parent, is_thread=is_thread)
        debug("Attach %s to debugger" % process)
        self.dict[pid] = process
        self.list.append(process)
        try:
            process.waitSignals(SIGTRAP, SIGSTOP)
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
        debug("Quit debugger")
        # Terminate processes in reverse order
        # to kill children before parents
        processes = list(self.list)
        for process in reversed(processes):
            process.terminate()
            process.detach()

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

    def _wait_event(self, wanted_pid, blocking=True):
        """
        Wait for a process event from the specified process identifier. If
        blocking=False, return None if there is no new event, otherwise return
        an object based on ProcessEvent.
        """
        process = None
        while not process:
            try:
                if wanted_pid and wanted_pid in self.dict and \
                        wanted_pid in self.sig_queue and self.sig_queue[wanted_pid]:
                    warning(f"waitpid(): Fetching signal for PID {wanted_pid} from queue")
                    pid, status = (wanted_pid, self.sig_queue[wanted_pid].pop(0))
                    if not self.sig_queue[wanted_pid]:
                        self.sig_queue.pop(wanted_pid)
                else:
                    pid, status = self._waitpid(wanted_pid, blocking=blocking)
            except OSError as err:
                if err.errno == ECHILD:
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
                        if stat.ppid in self.dict:
                            warning(f"Received premature signal for a child ({pid=}) of a traced process")
                        else:
                            enq = False
                            debug(f"Received signal for unknown {pid=}, ignoring")
                    except ProcError:
                        debug(f"Process ({pid=}) died before its signal could be processed")
                        enq = False
                else:
                    warning(f"Received signal for unknown {pid=}, placing event in queue")
                if enq:
                    if pid not in self.sig_queue:
                        self.sig_queue[pid] = list()
                    self.sig_queue[pid].append(status)
                continue

        return process.processStatus(status)

    def waitProcessEvent(self, pid=None, blocking=True):
        """
        Wait for a process event from a specific process (if pid option is
        set) or any process (default). If blocking=False, return None if there
        is no new event, otherwise return an object based on ProcessEvent.
        """
        return self._wait_event(pid, blocking=blocking)

    def waitSignals(self, *signals, blocking=True, **kw):
        """
        Wait for any signal or some specific signals (if specified) from a
        specific process (if pid keyword is set) or any process (default).
        Return a ProcessSignal object or raise an unexpected ProcessEvent.
        """
        pid = kw.get('pid', None)
        event = self._wait_event(pid, blocking=blocking)
        if event is None:
            return
        if event.__class__ != ProcessSignal:
            raise event
        signum = event.signum
        if signum in signals or not signals:
            return event
        raise event

    def waitSyscall(self, process=None, blocking=True):
        """
        Wait for the next syscall event (enter or exit) for a specific process
        (if specified) or any process (default). Return a ProcessSignal object
        or raise an unexpected ProcessEvent.
        """
        signum = SIGTRAP
        if self.use_sysgood:
            signum |= 0x80
        if process:
            return self.waitSignals(signum, blocking=blocking, pid=process.pid)
        else:
            return self.waitSignals(signum, blocking=blocking)

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

    def traceFork(self):
        """
        Enable fork() tracing. Do nothing if it's not supported.
        """
        if not HAS_PTRACE_EVENTS:
            raise DebuggerError(
                "Tracing fork events is not supported on this architecture or operating system")
        self.options |= PTRACE_O_TRACEFORK | PTRACE_O_TRACEVFORK
        self.trace_fork = True
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

    def __getitem__(self, pid):
        return self.dict[pid]

    def __iter__(self):
        return iter(self.list)

    def __len__(self):
        return len(self.list)
