from tango.ptrace.syscall import PtraceSyscall

from signal import SIGTRAP

class SyscallState(object):

    def __init__(self, process):
        self.process = process
        self.ignore_exec_trap = True
        self.ignore_callback = None
        self.clear()

    async def event(self, options):
        if self.next_event == "exit":
            return await self.exit()
        else:
            return await self.enter(options)

    async def enter(self, options):
        # syscall enter
        regs = self.process.getregs()
        self.syscall = PtraceSyscall(self.process, options, regs)
        self.name = self.syscall.name
        if (not self.ignore_callback) \
                or (not self.ignore_callback(self.syscall)):
            self.syscall.enter(regs)
        else:
            self.syscall = None
        self.next_event = "exit"
        return self.syscall

    async def exit(self):
        if self.syscall:
            self.syscall.exit()
        if self.ignore_exec_trap \
                and self.name == "execve" \
                and not self.process.debugger.trace_exec:
            # Ignore the SIGTRAP after exec() syscall exit
            self.process.syscall()
            await self.process.waitSignals(SIGTRAP, SIGTRAP | 0x80)
        if self.syscall and ((not self.ignore_callback) \
                or (not self.ignore_callback(self.syscall))):
            syscall = self.syscall
        else:
            syscall = None
        self.clear()
        return syscall

    def clear(self):
        self.syscall = None
        self.name = None
        self.next_event = "enter"
