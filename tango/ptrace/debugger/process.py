from tango.ptrace import debug, info, warning, critical, error
from tango.ptrace.binding import (
    HAS_PTRACE_SINGLESTEP, HAS_PTRACE_EVENTS,
    HAS_PTRACE_SIGINFO, HAS_PTRACE_IO, HAS_PTRACE_GETREGS,
    HAS_PTRACE_GETREGSET,
    ptrace_attach, ptrace_detach,
    ptrace_cont, ptrace_syscall,
    ptrace_setregs,
    ptrace_peektext, ptrace_poketext,
    REGISTER_NAMES)
from tango.ptrace.os_tools import HAS_PROC, RUNNING_BSD, RUNNING_PYTHON3
from tango.ptrace.tools import dumpRegs
from tango.ptrace.cpu_info import CPU_WORD_SIZE
from tango.ptrace.ctypes_tools import bytes2word, word2bytes, bytes2type, bytes2array
from tango.ptrace.ctypes_tools import formatAddress, formatWordHex
from tango.ptrace.errors import PtraceError
from tango.ptrace.debugger import (Breakpoint, ProcessExit, ProcessSignal,
    NewProcessEvent, ProcessExecution, SeccompEvent)
from tango.ptrace.disasm import HAS_DISASSEMBLER
from tango.ptrace.debugger.backtrace import getBacktrace
from tango.ptrace.debugger.process_error import ProcessError
from tango.ptrace.debugger.memory_mapping import readProcessMappings
from tango.ptrace.binding.cpu import CPU_INSTR_POINTER, CPU_STACK_POINTER, CPU_FRAME_POINTER, CPU_SUB_REGISTERS
from tango.ptrace.debugger.syscall_state import SyscallState
if HAS_PTRACE_SINGLESTEP:
    from tango.ptrace.binding import ptrace_singlestep
if HAS_PTRACE_SIGINFO:
    from tango.ptrace.binding import ptrace_getsiginfo
if HAS_PTRACE_IO:
    from ctypes import create_string_buffer, addressof
    from tango.ptrace.binding import (
        ptrace_io, ptrace_io_desc,
        PIOD_READ_D, PIOD_WRITE_D)
if HAS_PTRACE_EVENTS:
    from tango.ptrace.binding import (
        ptrace_setoptions, ptrace_geteventmsg, WPTRACEEVENT,
        PTRACE_EVENT_FORK, PTRACE_EVENT_VFORK, PTRACE_EVENT_CLONE,
        PTRACE_EVENT_EXEC, PTRACE_EVENT_SECCOMP)
    NEW_PROCESS_EVENT = (
        PTRACE_EVENT_FORK, PTRACE_EVENT_VFORK, PTRACE_EVENT_CLONE)
if HAS_PTRACE_GETREGS or HAS_PTRACE_GETREGSET:
    from tango.ptrace.binding import ptrace_getregs
else:
    from tango.ptrace.binding import ptrace_peekuser, ptrace_registers_t
if HAS_DISASSEMBLER:
    from tango.ptrace.disasm import disassemble, disassembleOne, MAX_INSTR_SIZE
if HAS_PROC:
    from tango.ptrace.linux_proc import readProcessStat

from ctypes import sizeof, c_char_p
from errno import ESRCH, EACCES
from signal import SIGTRAP, SIGSTOP, SIGKILL, SIGTERM
from os import (kill,
                WIFSTOPPED, WSTOPSIG,
                WIFSIGNALED, WTERMSIG,
                WIFEXITED, WEXITSTATUS)
from contextlib import contextmanager

MIN_CODE_SIZE = 32
MAX_CODE_SIZE = 1024
DEFAULT_NB_INSTR = 10
DEFAULT_CODE_SIZE = 24

class PtraceProcess(object):
    """
    Process traced by a PtraceDebugger.

    Methods
    =======

     * control execution:

       - singleStep(): execute one instruction
       - cont(): continue the execution
       - syscall(): break at next syscall
       - setInstrPointer(): change the instruction pointer
       - kill(): send a signal to the process
       - terminate(): kill the process

     * wait an event:

      - waitEvent(): wait next process event
      - waitSignals(): wait a signal

     * get status

       - getreg(): get a register
       - getInstrPointer(): get the instruction pointer
       - getStackPointer(): get the stack pointer
       - getFramePointer(): get the stack pointer
       - getregs(): get all registers, e.g. regs=getregs(); print regs.eax
       - disassemble(): assembler code of the next instructions
       - disassembleOne(): assembler code of the next instruction
       - findStack(): get stack memory mapping
       - getsiginfo(): get signal information
       - getBacktrace(): get the current backtrace

     * set status

       - setreg(): set a register
       - setregs(): set all registers

     * memory access:

       - readWord(): read a memory word
       - readBytes(): read some bytes
       - readStruct(): read a structure
       - readArray(): read an array
       - readCString(): read a C string
       - readMappings(): get all memory mappings
       - writeWord(): write a memory word
       - writeBytes(): write some bytes

     * display status:

       - dumpCode(): display the next instructions
       - dumpStack(): display some memory words around the stack pointer
       - dumpMaps(): display memory mappings
       - dumpRegs(): display all registers

     * breakpoint:

       - createBreakpoint(): set a breakpoint
       - findBreakpoint(): find a breakpoint
       - removeBreakpoint(): remove a breakpoint

     * other:

       - setoptions(): set ptrace options

    See each method to get better documentation. You are responsible
    to manage the process state: some methods may fail or crash your
    processus if they are called when the process is in the wrong
    state.

    Attributes
    ==========

     * main attributes:
       - pid: identifier of the process
       - debugger: PtraceDebugger instance
       - breakpoints: dictionary of active breakpoints
       - parent: parent PtraceProcess (None if process has no parent)

     * state:
       - running: if True, the process is alive, otherwise the process
         doesn't exist anymore
       - exited: if True, the process has exited (attributed only used
         on BSD operation systems)
       - is_attached: if True, the process is attached by ptrace
       - was_attached: if True, the process will be detached at exit
       - is_stopped: if True, the process is stopped, otherwise it's
         running
       - syscall_state: control syscall tracing

    Sometimes, is_stopped value is wrong. You might use isTraced() to
    make sure that the process is stopped.
    """

    def __init__(self, debugger, pid, is_attached, parent=None, is_thread=False,
            subscription=None, owns_subscription=False):
        self.debugger = debugger
        self.breakpoints = {}
        self.pid = pid
        self.running = True
        self.exited = False
        self.parent = parent
        self.subscription = subscription
        self.owns_subscription = owns_subscription
        if not subscription and parent:
            self.subscription = parent.subscription
            self.owns_subscription = False
        if parent and parent.root:
            self.root = parent.root
        else:
            self.root = self
        self.children = []
        self.was_attached = is_attached
        self.is_attached = False
        self.is_stopped = True
        self.is_thread = is_thread
        if not is_attached:
            self.attach()
        else:
            self.is_attached = True
        if HAS_PROC:
            self.read_mem_file = None
        self.syscall_state = SyscallState(self)

        self._ctx = None

    def isTraced(self):
        if not HAS_PROC:
            self.notImplementedError()
        stat = readProcessStat(self.pid)
        return (stat.state in ('t', 'T'))

    def attach(self):
        if self.is_attached:
            return
        debug("Attach process %s" % self.pid)
        ptrace_attach(self.pid)
        self.is_attached = True

    def dumpCode(self, start=None, stop=None, manage_bp=False, log=None):
        if not log:
            log = error
        try:
            ip = self.getInstrPointer()
        except PtraceError as err:
            if start is None:
                log("Unable to read instruction pointer: %s" % err)
                return
            ip = None
        if start is None:
            start = ip

        try:
            self._dumpCode(start, stop, ip, manage_bp, log)
        except PtraceError as err:
            log("Unable to dump code at %s: %s" % (
                formatAddress(start), err))

    def _dumpCode(self, start, stop, ip, manage_bp, log):
        if stop is not None:
            stop = max(start, stop)
            stop = min(stop, start + MAX_CODE_SIZE - 1)

        if not HAS_DISASSEMBLER:
            if stop is not None:
                size = stop - start + 1
            else:
                size = MIN_CODE_SIZE
            code = self.readBytes(start, size)
            if RUNNING_PYTHON3:
                text = " ".join("%02x" % byte for byte in code)
            else:
                text = " ".join("%02x" % ord(byte) for byte in code)
            log("CODE: %s" % text)
            return

        log("CODE:")
        if manage_bp:
            address = start
            for line in range(10):
                bp = False
                if address in self.breakpoints:
                    bytes = self.breakpoints[address].old_bytes
                    instr = disassembleOne(bytes, address)
                    bp = True
                else:
                    instr = self.disassembleOne(address)
                text = "%s| %s (%s)" % (formatAddress(
                    instr.address), instr.text, instr.hexa)
                if instr.address == ip:
                    text += " <=="
                if bp:
                    text += "     * BREAKPOINT *"
                log(text)
                address = address + instr.size
                if stop is not None and stop <= address:
                    break
        else:
            for instr in self.disassemble(start, stop):
                text = "%s| %s (%s)" % (formatAddress(
                    instr.address), instr.text, instr.hexa)
                if instr.address == ip:
                    text += " <=="
                log(text)

    def disassemble(self, start=None, stop=None, nb_instr=None):
        if not HAS_DISASSEMBLER:
            self.notImplementedError()
        if start is None:
            start = self.getInstrPointer()
        if stop is not None:
            stop = max(start, stop)
            size = stop - start + 1
        else:
            if nb_instr is None:
                nb_instr = DEFAULT_NB_INSTR
            size = nb_instr * MAX_INSTR_SIZE

        code = self.readBytes(start, size)
        for index, instr in enumerate(disassemble(code, start)):
            yield instr
            if nb_instr and nb_instr <= (index + 1):
                break

    def disassembleOne(self, address=None):
        if not HAS_DISASSEMBLER:
            self.notImplementedError()
        if address is None:
            address = self.getInstrPointer()
        code = self.readBytes(address, MAX_INSTR_SIZE)
        return disassembleOne(code, address)

    def findStack(self):
        for map in self.readMappings():
            if map.pathname == "[stack]":
                return map
        return None

    def detach(self):
        if not self.is_attached:
            return
        self.is_attached = False
        try:
            if self.running:
                debug("Detach %s" % self)
                ptrace_detach(self.pid)
        finally:
            self.deleteFromDebugger()

    def deleteFromDebugger(self):
        if self.subscription and self.owns_subscription:
            self.debugger.unsubscribe(self.subscription)
            self.subscription = None

        # remove self from parent
        if self.parent is not None and self in self.parent.children:
            self.parent.children.remove(self)
            self.parent = None

        self.debugger.deleteProcess(process=self)

    def _notRunning(self):
        if HAS_PROC and self.read_mem_file:
            try:
                self.read_mem_file.close()
            except IOError:
                pass
        # FIXME the order of these may need to be flipped
        self.running = False
        self.detach()

    def kill(self, signum):
        try:
            kill(self.pid, signum)
        except ProcessLookupError as ex:
            warning(f"kill({self.pid}): Process doesn't exist")
            return False
        return True

    async def terminate(self, wait_exit=True, signum=SIGTERM):
        if not self.running or not self.was_attached:
            return True
        debug("Terminate %s" % self)
        done = False
        try:
            if self.is_stopped:
                self.cont(signum)
            else:
                if not self.kill(signum):
                    raise PtraceError("No such process",
                        errno=ESRCH, pid=self.pid)
        except PtraceError as event:
            if event.errno == ESRCH:
                done = True
            else:
                raise event
        if not done:
            if not wait_exit:
                return False
            await self.waitExit()
        self._notRunning()
        return True

    async def terminateTree(self, **kwargs):
        try:
            while True:
                try:
                    # WARN it seems necessary to wait for the child to exit,
                    # otherwise the forkserver may misbehave, and the fuzzer
                    # will receive a lot of ForkChildKilledEvents
                    await self.terminate(**kwargs)
                    break
                except PtraceError as ex:
                    critical(f"Attempted to terminate non-existent process ({ex})")
                    break
                except ProcessExit as ex:
                    debug(f"{ex.process} exited while terminating {self}")
                    continue
        finally:
            if self in self.debugger:
                self.deleteFromDebugger()
        for p in self.children:
            await p.terminateTree(**kwargs)

    async def waitExit(self):
        while True:
            # Wait for any process signal
            event = await self.waitEvent()
            event_cls = event.__class__

            # Process exited: we are done
            if event_cls == ProcessExit and event.process == self:
                return

            try:
                if event.is_syscall_stop():
                    event.process.cont()
                elif event_cls is ProcessSignal:
                    # Send the signal to the process
                    signum = event.signum
                    if event.signum not in (SIGTRAP, SIGSTOP):
                        event.process.cont(event.signum)
                    else:
                        event.process.cont()
                elif event_cls is ProcessExecution:
                    warning(f"{event.process} called exec() while terminating")
                else:
                    # Event different than a signal? Raise an exception
                    raise event
            except PtraceError as ex:
                debug(f"{event.process} received {event} while waiting for exit,"
                    f" but the signal could not be delivered {ex=}.")

    async def processStatus(self, status):
        # Process exited?
        if WIFEXITED(status):
            code = WEXITSTATUS(status)
            event = await self.processExited(code)

        # Process killed by a signal?
        elif WIFSIGNALED(status):
            signum = WTERMSIG(status)
            event = await self.processKilled(signum)

        # Invalid process status?
        elif not WIFSTOPPED(status):
            raise ProcessError(self, "Unknown process status: %r" % status)

        # Ptrace event?
        elif HAS_PTRACE_EVENTS and WPTRACEEVENT(status):
            event = WPTRACEEVENT(status)
            event = await self.ptraceEvent(event)

        else:
            signum = WSTOPSIG(status)
            event = await self.processSignal(signum)
        return event

    async def processTerminated(self):
        self._notRunning()
        return ProcessExit(self)

    async def processExited(self, code):
        if RUNNING_BSD and not self.exited:
            # on FreeBSD, we have to waitpid() twice
            # to avoid zombi process!?
            self.exited = True
            await self.waitExit()
        self._notRunning()
        return ProcessExit(self, exitcode=code)

    async def processKilled(self, signum):
        self._notRunning()
        return ProcessExit(self, signum=signum)

    async def processSignal(self, signum):
        self.is_stopped = True
        return ProcessSignal(signum, self)

    async def processSeccompEvent(self, event):
        self.is_stopped = True
        if self.syscall_state.next_event == 'exit':
            # tracer is aware of syscall-entry, so we must not double-notify it.
            # instead, we'll resume the process until syscall-exit occurs
            debug("seccomp_trace event received while expecting"
                  " syscall-exit-stop; retrying")
            self.syscall()
            return await self.waitEvent()
        else:
            debug("seccomp_trace event received as syscall-enter-stop")
            return event

    async def ptraceEvent(self, event):
        if not HAS_PTRACE_EVENTS:
            self.notImplementedError()
        if event in NEW_PROCESS_EVENT:
            new_pid = ptrace_geteventmsg(self.pid)
            is_thread = (event == PTRACE_EVENT_CLONE)
            new_process = await self.debugger.addProcess(
                new_pid, is_attached=True, parent=self, is_thread=is_thread)
            self.children.append(new_process)
            return NewProcessEvent(new_process)
        elif event == PTRACE_EVENT_EXEC:
            return ProcessExecution(self)
        elif event == PTRACE_EVENT_SECCOMP:
            event = SeccompEvent(self)
            return await self.processSeccompEvent(event)
        else:
            raise ProcessError(self, "Unknown ptrace event: %r" % event)

    @contextmanager
    def regsctx(self):
        # WARN not compatible with asyncio or threads
        if self._ctx:
            yield
        else:
            self._ctx = self.getregs()
            try:
                yield
            finally:
                if self._ctx:
                    self.setregs(self._ctx)
                    self._ctx = None

    def getregs(self):
        if HAS_PTRACE_GETREGS or HAS_PTRACE_GETREGSET:
            return ptrace_getregs(self.pid)
        else:
            # FIXME: Optimize getreg() when used with this function
            words = []
            nb_words = sizeof(ptrace_registers_t) // CPU_WORD_SIZE
            for offset in range(nb_words):
                word = ptrace_peekuser(self.pid, offset * CPU_WORD_SIZE)
                bytes = word2bytes(word)
                words.append(bytes)
            bytes = ''.join(words)
            return bytes2type(bytes, ptrace_registers_t)

    def queryregs(self, regs, name):
        try:
            name, shift, mask = CPU_SUB_REGISTERS[name]
        except KeyError:
            shift = 0
            mask = None
        if name not in REGISTER_NAMES:
            raise ProcessError(self, "Unknown register: %r" % name)
        value = getattr(regs, name)
        value >>= shift
        if mask:
            value &= mask
        return value

    def getreg(self, name):
        if self._ctx:
            regs = self._ctx
        else:
            regs = self.getregs()
        return self.queryregs(regs, name)

    def setregs(self, regs):
        ptrace_setregs(self.pid, regs)

    def updateregs(self, regs, name, value):
        if name in CPU_SUB_REGISTERS:
            full_name, shift, mask = CPU_SUB_REGISTERS[name]
            full_value = getattr(regs, full_name)
            full_value &= ~mask
            full_value |= ((value & mask) << shift)
            value = full_value
            name = full_name
        if name not in REGISTER_NAMES:
            raise ProcessError(self, "Unknown register: %r" % name)
        setattr(regs, name, value)

    def setreg(self, name, value):
        if self._ctx:
            self.updateregs(self._ctx, name, value)
        else:
            regs = self.getregs()
            self.updateregs(regs, name, value)
            self.setregs(regs)

    def singleStep(self):
        if not HAS_PTRACE_SINGLESTEP:
            self.notImplementedError()
        ptrace_singlestep(self.pid)

    def filterSignal(self, signum):
        if self.debugger.use_sysgood:
            signum &= 0x7F
        if signum == SIGTRAP:
            # Never transfer SIGTRAP signal
            return 0
        else:
            return signum

    def _flush_ctx(self):
        if self._ctx:
            # flush context before resuming target
            self.setregs(self._ctx)
            self._ctx = None

    def syscall(self, signum=0):
        self._flush_ctx()
        signum = self.filterSignal(signum)
        ptrace_syscall(self.pid, signum)
        self.is_stopped = False

    def setInstrPointer(self, ip):
        if CPU_INSTR_POINTER:
            self.setreg(CPU_INSTR_POINTER, ip)
        else:
            raise ProcessError(
                self, "Instruction pointer register is not defined")

    def getInstrPointer(self):
        if CPU_INSTR_POINTER:
            return self.getreg(CPU_INSTR_POINTER)
        else:
            raise ProcessError(
                self, "Instruction pointer register is not defined")

    def getStackPointer(self):
        if CPU_STACK_POINTER:
            return self.getreg(CPU_STACK_POINTER)
        else:
            raise ProcessError(self, "Stack pointer register is not defined")

    def getFramePointer(self):
        if CPU_FRAME_POINTER:
            return self.getreg(CPU_FRAME_POINTER)
        else:
            raise ProcessError(self, "Stack pointer register is not defined")

    def _readBytes(self, address, size):
        offset = address % CPU_WORD_SIZE
        if offset:
            # Read word
            address -= offset
            word = self.readWord(address)
            bytes = word2bytes(word)

            # Read some bytes from the word
            subsize = min(CPU_WORD_SIZE - offset, size)
            data = bytes[offset:offset + subsize]   # <-- FIXME: Big endian!

            # Move cursor
            size -= subsize
            address += CPU_WORD_SIZE
        else:
            data = b''

        while size:
            # Read word
            word = self.readWord(address)
            bytes = word2bytes(word)

            # Read bytes from the word
            if size < CPU_WORD_SIZE:
                data += bytes[:size]   # <-- FIXME: Big endian!
                break
            data += bytes

            # Move cursor
            size -= CPU_WORD_SIZE
            address += CPU_WORD_SIZE
        return data

    def readWord(self, address):
        """Address have to be aligned!"""
        word = ptrace_peektext(self.pid, address)
        return word

    if HAS_PTRACE_IO:
        def readBytes(self, address, size):
            buffer = create_string_buffer(size)
            io_desc = ptrace_io_desc(
                piod_op=PIOD_READ_D,
                piod_offs=address,
                piod_addr=addressof(buffer),
                piod_len=size)
            ptrace_io(self.pid, io_desc)
            return buffer.raw
    elif HAS_PROC:
        def readBytes(self, address, size):
            if not self.read_mem_file:
                filename = '/proc/%u/mem' % self.pid
                try:
                    self.read_mem_file = open(filename, 'rb', 0)
                except IOError as err:
                    message = "Unable to open %s: fallback to ptrace implementation" % filename
                    if err.errno != EACCES:
                        error(message)
                    else:
                        warn(message)
                    self.readBytes = self._readBytes
                    return self.readBytes(address, size)

            try:
                mem = self.read_mem_file
                mem.seek(address)
                data = mem.read(size)
            except (IOError, ValueError) as err:
                raise ProcessError(self, "readBytes(%s, %s) error: %s" % (
                    formatAddress(address), size, err))
            if len(data) == 0 and size:
                # Issue #10: If the process was not created by the debugger
                # (ex: fork), the kernel may deny reading private mappings of
                # /proc/pid/mem to the debugger, depending on the kernel
                # version and kernel config (ex: SELinux enabled or not).
                #
                # Fallback to PTRACE_PEEKTEXT. It is slower but a debugger
                # tracing the process is always allowed to use it.
                self.readBytes = self._readBytes
                return self.readBytes(address, size)
            return data
    else:
        readBytes = _readBytes

    def getsiginfo(self):
        if not HAS_PTRACE_SIGINFO:
            self.notImplementedError()
        return ptrace_getsiginfo(self.pid)

    def writeBytes(self, address, bytes):
        if HAS_PTRACE_IO:
            size = len(bytes)
            bytes = create_string_buffer(bytes)
            io_desc = ptrace_io_desc(
                piod_op=PIOD_WRITE_D,
                piod_offs=address,
                piod_addr=addressof(bytes),
                piod_len=size)
            ptrace_io(self.pid, io_desc)
        else:
            offset = address % CPU_WORD_SIZE
            if offset:
                # Write partial word (end)
                address -= offset
                size = CPU_WORD_SIZE - offset
                word = self.readBytes(address, CPU_WORD_SIZE)
                if len(bytes) < size:
                    size = len(bytes)
                    word = word[:offset] + bytes[:size] + \
                        word[offset + size:]  # <-- FIXME: Big endian!
                else:
                    # <-- FIXME: Big endian!
                    word = word[:offset] + bytes[:size]
                self.writeWord(address, bytes2word(word))
                bytes = bytes[size:]
                address += CPU_WORD_SIZE

            # Write full words
            while CPU_WORD_SIZE <= len(bytes):
                # Read one word
                word = bytes[:CPU_WORD_SIZE]
                word = bytes2word(word)
                self.writeWord(address, word)

                # Move to next word
                bytes = bytes[CPU_WORD_SIZE:]
                address += CPU_WORD_SIZE
            if not bytes:
                return

            # Write partial word (begin)
            size = len(bytes)
            word = self.readBytes(address, CPU_WORD_SIZE)
            # FIXME: Write big endian version of the next line
            word = bytes + word[size:]
            self.writeWord(address, bytes2word(word))

    def readStruct(self, address, struct):
        bytes = self.readBytes(address, sizeof(struct))
        bytes = c_char_p(bytes)
        return bytes2type(bytes, struct)

    def readArray(self, address, basetype, count):
        bytes = self.readBytes(address, sizeof(basetype) * count)
        bytes = c_char_p(bytes)
        return bytes2array(bytes, basetype, count)

    def readCString(self, address, max_size, chunk_length=256):
        string = []
        size = 0
        truncated = False
        while True:
            done = False
            data = self.readBytes(address, chunk_length)
            pos = data.find(b'\0')
            if pos != -1:
                done = True
                data = data[:pos]
            if max_size <= size + chunk_length:
                data = data[:(max_size - size)]
                string.append(data)
                truncated = True
                break
            string.append(data)
            if done:
                break
            size += chunk_length
            address += chunk_length
        return b''.join(string), truncated

    def dumpStack(self, log=None):
        if not log:
            log = error
        stack = self.findStack()
        if stack:
            log("STACK: %s" % stack)
        self._dumpStack(log)

    def _dumpStack(self, log):
        sp = self.getStackPointer()
        displayed = 0
        for index in range(-5, 5 + 1):
            delta = index * CPU_WORD_SIZE
            try:
                value = self.readWord(sp + delta)
                log("STACK%+ 3i: %s" % (delta, formatWordHex(value)))
                displayed += 1
            except PtraceError:
                pass
        if not displayed:
            log("ERROR: unable to read the stack (SP=%s)" % formatAddress(sp))

    def readMappings(self):
        return readProcessMappings(self)

    def dumpMaps(self, log=None):
        if not log:
            log = error
        for map in self.readMappings():
            log("MAPS: %s" % map)

    def writeWord(self, address, word):
        """
        Address have to be aligned!
        """
        ptrace_poketext(self.pid, address, word)

    def dumpRegs(self, log=None):
        if not log:
            log = error
        try:
            regs = self.getregs()
            dumpRegs(log, regs)
        except PtraceError as err:
            log("Unable to read registers: %s" % err)

    def cont(self, signum=0):
        self._flush_ctx()
        signum = self.filterSignal(signum)
        ptrace_cont(self.pid, signum)
        self.is_stopped = False

    def setoptions(self, options):
        if not HAS_PTRACE_EVENTS:
            self.notImplementedError()
        debug("Set %s options to %s" % (self, options))
        ptrace_setoptions(self.pid, options)

    async def waitEvent(self, blocking=True):
        return await self.debugger.waitProcessEvent(
            process=self, blocking=blocking, subscription=self.subscription)

    async def waitSignals(self, *signals, blocking=True):
        return await self.debugger.waitSignals(*signals,
            process=self, blocking=blocking, subscription=self.subscription)

    async def waitSyscall(self, blocking=True):
        return await self.debugger.waitSyscall(self,
            blocking=blocking, subscription=self.subscription)

    def findBreakpoint(self, address):
        for bp in self.breakpoints.values():
            if bp.address <= address < bp.address + bp.size:
                return bp
        return None

    def createBreakpoint(self, address, size=1):
        bp = self.findBreakpoint(address)
        if bp:
            raise ProcessError(self, "A breakpoint is already set: %s" % bp)
        bp = Breakpoint(self, address, size)
        self.breakpoints[address] = bp
        return bp

    def getBacktrace(self, max_args=6, max_depth=20):
        return getBacktrace(self, max_args=max_args, max_depth=max_depth)

    def removeBreakpoint(self, breakpoint):
        del self.breakpoints[breakpoint.address]

    def setSubscription(self, subscription, owned=False):
        self.subscription = subscription
        self.owns_subscription = owned

    def forkSubscription(self):
        if not self.subscription:
            raise RuntimeError("No subscription specified")
        self.subscription = self.subscription.fork(wanted_pid=self.pid)
        self.owns_subscription = True

    def __del__(self):
        try:
            self.detach()
        except PtraceError:
            pass

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<PtraceProcess #%s>" % self.pid

    def __hash__(self):
        return hash(self.pid)

    def notImplementedError(self):
        raise NotImplementedError
