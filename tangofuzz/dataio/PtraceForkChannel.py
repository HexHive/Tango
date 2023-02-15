from . import debug, warning, info, error

from abc import abstractmethod
from dataio import PtraceChannel
from   common      import ProcessCrashedException
from profiler import ProfileCount
from ptrace import PtraceError
from ptrace.debugger import   (PtraceProcess,
                               ProcessEvent,
                               NewProcessEvent,
                               ForkChildKilledEvent)
from ptrace.binding.cpu import CPU_INSTR_POINTER, CPU_STACK_POINTER
from subprocess import Popen
import signal
import sys
from elftools.elf.elffile import ELFFile
from ptrace.cpu_info import CPU_WORD_SIZE

class PtraceForkChannel(PtraceChannel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._proc_trapped = False
        self._proc_untrap = False
        self._wait_for_proc = False
        self._event_queue = []
        self._forked_child = None

        # extract forkserver location
        with open(self._pobj.args[0], 'rb') as f:
            elf = ELFFile(f)

            # get forkserver offset from image base
            if not elf.has_dwarf_info():
                raise RuntimeError("Debug symbols are needed for forkserver")
            dwarf = elf.get_dwarf_info()
            try:
                debug("Searching for forkserver symbol")
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
                assert vmmaps[0].pathname == self._pobj.args[0], \
                    ("Path to first module in the vmmap did not match the path"
                     " to the executable. Maybe you are using a symbolic link"
                     " as the path to the target?")
                run_base = vmmaps[0].start
                self._forkserver = base_addr + run_base
            debug(f"Forkserver found at 0x{self._forkserver:016X}")

    def process_exit(self, event):
        try:
            super().process_exit(event)
        finally:
            # if the current target exits unexpectedly, we also report it to the forkserver
            if event.process == self._forked_child:
                self._wakeup_forkserver()
            elif event.process == self._proc:
                error("Forkserver crashed!")
                raise ForkserverCrashedException()

    def process_signal(self, event):
        if event.signum == signal.SIGTRAP:
            # If the forkserver traps, that means it's waiting for its child to
            # die. We will wake it up when we kill the forked_child.
            # Otherwise, the child trapped and we resume its execution
            if event.process != self._proc:
                self._forked_child = event.process
                # restore correct trap byte and registers
                self._cleanup_forkserver(event.process)
                # resume execution of the child process
                event.process.syscall()
            else:
                debug("Forkserver trapped, waiting for wake-up call")
                self._proc_trapped = True
                if self._proc_untrap:
                    # When the child dies unexpectedly (ForkChildKilledEvent),
                    # we wake up the server
                    self._wakeup_forkserver()
                    self._proc_untrap = False
                self._wait_for_proc = False
        elif event.signum == signal.SIGCHLD and event.process == self._proc:
            # when the forkserver receives SIGCHLD, we ignore it
            event.process.syscall()
        else:
            super().process_signal(event)

    def process_auxiliary_event(self, event, ignore_callback):
        try:
            raise event
        except NewProcessEvent as event:
            self.process_new(event, ignore_callback)
            if event.process.parent == self._proc and event.process.is_attached:
                # pause children and queue up syscalls until forkserver traps
                self._wait_for_proc = True
        except ForkChildKilledEvent as event:
            # this is sent by the debugger when the child dies unexpectedly (custom behavior)
            # FIXME check if there are exceptions to this
            # We need to resume the forkserver because addProcess raised an exception instead
            # of NewProcessEvent (so the forkserver is still stuck in the fork syscall)
            warning("Forked child died on entry, forkserver will be woken up")
            self._proc.syscall()
            self._proc_untrap = True
            ProfileCount("infant_mortality")(1)
        except Exception as event:
            super().process_auxiliary_event(event, ignore_callback)

    def _wait_for_syscall(self, process: PtraceProcess=None):
        # this next block ensures that a forked child does not exit before
        # the forkserver traps. in that scenario, the wake-up call is sent
        # before the forkserver traps, and then it traps forever
        event = None
        if not self._wait_for_proc and self._event_queue:
            event = self._event_queue.pop(0)
        else:
            event = self._debugger.waitSyscall(process=process)
            if self._wait_for_proc and event.process != self._proc:
                self._event_queue.append(event)
                debug("Received event while waiting for forkserver; enqueued!")
                event = None
        return event

    def _stack_push(self, process, value):
        rsp = process.getStackPointer() - 8
        process.writeBytes(rsp, value.to_bytes(8, byteorder=sys.byteorder))
        process.setreg(CPU_STACK_POINTER, rsp)

    def _stack_pop(self, process):
        rsp = process.getStackPointer()
        value = int.from_bytes(process.readBytes(rsp, 8), byteorder=sys.byteorder)
        process.setreg(CPU_STACK_POINTER, rsp + 8)
        return value

    def _inject_forkserver(self, process: PtraceProcess, address: int):
        debug("Injecting forkserver!")
        with process.regsctx():
            self._trap_rip = address # we need this to restore execution later
            word_offset = address % CPU_WORD_SIZE
            self._trap_rip_aligned = address - word_offset
            self._trap_rsp = process.getStackPointer()

            # read the original word where a byte needs be replaced by a trap
            self._trap_asm_word = process.readWord(self._trap_rip_aligned)

            # place a trap
            trap_mask = 0xff << word_offset
            trap_word = (self._trap_asm_word & ~trap_mask) | (0xCC << word_offset)
            process.writeWord(self._trap_rip_aligned, trap_word)
            process.setreg(CPU_STACK_POINTER, self._trap_rsp & ~0x0F)

            # set up the stack, so that it returns to the trap
            self._stack_push(process, 0) # some x86-64 alignment stuff
            self._stack_push(process, address)

            # redirect control flow to the forkserver
            process.setreg(CPU_INSTR_POINTER, self._forkserver)

    def _invoke_forkserver(self, process: PtraceProcess):
        address = process.getInstrPointer()
        self._inject_forkserver(process, address)

    def _cleanup_forkserver(self, process: PtraceProcess):
        with process.regsctx():
            # restore the original byte that was replaced by the trap
            process.writeWord(self._trap_rip_aligned, self._trap_asm_word)
            # restore the stack pointer to its proper location
            process.setreg(CPU_STACK_POINTER, self._trap_rsp)
            # redirect control flow back where it should have resumed
            process.setreg(CPU_INSTR_POINTER, self._trap_rip)

    def _wakeup_forkserver(self):
        if self._proc_trapped:
            debug("Waking up forkserver :)")
            self._proc.syscall()

            # must actually wait for syscall, not any event
            self._wakeup_forkserver_syscall_found = False

            # backup the old ignore_callbacks
            for process in self._debugger:
                process.syscall_state._ignore_callback = process.syscall_state.ignore_callback
            self.monitor_syscalls(None, \
                self._wakeup_forkserver_ignore_callback, \
                self._wakeup_forkserver_break_callback, \
                self._wakeup_forkserver_syscall_callback, break_on_entry=True)
            # restore the old ignore_callbacks
            for process in self._debugger:
                process.syscall_state.ignore_callback = process.syscall_state._ignore_callback
                del process.syscall_state._ignore_callback
            self._proc_trapped = False

    def close(self, terminate, **kwargs):
        if self._forked_child:
            if terminate:
                self.terminator(self._forked_child)
            # when we kill the forked_child, we wake up the forkserver from the trap
            self._wakeup_forkserver()

    ### Callbacks ###
    def _invoke_forkserver_ignore_callback(self, syscall):
        return True

    def _invoke_forkserver_break_callback(self):
        return self._proc_trapped

    def _invoke_forkserver_syscall_callback(self, process, syscall):
        process.syscall()

    def _wakeup_forkserver_ignore_callback(self, syscall):
        return False

    def _wakeup_forkserver_break_callback(self):
        return self._wakeup_forkserver_syscall_found

    def _wakeup_forkserver_syscall_callback(self, process, syscall):
        self._wakeup_forkserver_syscall_found = True
        process.syscall()

class ForkserverCrashedException(RuntimeError):
    pass
