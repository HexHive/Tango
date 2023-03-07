from tango.ptrace.debugger import ProcessEvent
from tango.ptrace.os_tools import RUNNING_LINUX
if RUNNING_LINUX:
    from tango.ptrace.binding.func import ptrace_get_syscall_info

class SeccompEvent(ProcessEvent):
    def __init__(self, process):
        super().__init__(process, 'seccomp_trace')

    def is_syscall_stop(self):
        return True

    @property
    def syscall_info(self):
        return ptrace_get_syscall_info(self.process.pid).seccomp