from .func import (   # noqa
    HAS_PTRACE_SINGLESTEP, HAS_PTRACE_EVENTS,
    HAS_PTRACE_IO, HAS_PTRACE_SIGINFO, HAS_PTRACE_GETREGS,
    HAS_PTRACE_GETREGSET, REGISTER_NAMES, HAS_SECCOMP_FILTER,
    ptrace_attach, ptrace_traceme,
    ptrace_detach, ptrace_kill,
    ptrace_cont, ptrace_syscall,
    ptrace_setregs,
    ptrace_peektext, ptrace_poketext,
    ptrace_peekuser,
    ptrace_registers_t)
if HAS_PTRACE_EVENTS:
    from .func import (WPTRACEEVENT,   # noqa
                      PTRACE_EVENT_FORK, PTRACE_EVENT_VFORK, PTRACE_EVENT_CLONE,
                      PTRACE_EVENT_EXEC, PTRACE_O_TRACESECCOMP,
                      PTRACE_EVENT_SECCOMP,
                      ptrace_setoptions, ptrace_geteventmsg)
if HAS_PTRACE_SINGLESTEP:
    from .func import ptrace_singlestep   # noqa
if HAS_PTRACE_SIGINFO:
    from .func import ptrace_getsiginfo   # noqa
if HAS_PTRACE_IO:
    from .func import ptrace_io   # noqa
    from .freebsd_struct import (   # noqa
        ptrace_io_desc,
        PIOD_READ_D, PIOD_WRITE_D,
        PIOD_READ_I, PIOD_WRITE_I)
if HAS_PTRACE_GETREGS or HAS_PTRACE_GETREGSET:
    from .func import ptrace_getregs   # noqa

if HAS_SECCOMP_FILTER:
    from .func import (
        # bpf macros
        BPF_LD, BPF_LDX, BPF_ST, BPF_STX, BPF_ALU, BPF_JMP,
        BPF_RET, BPF_MISC, BPF_W, BPF_H, BPF_B, BPF_IMM, BPF_ABS, BPF_IND,
        BPF_MEM, BPF_LEN, BPF_MSH, BPF_ADD, BPF_SUB, BPF_MUL, BPF_DIV, BPF_OR,
        BPF_AND, BPF_LSH, BPF_RSH, BPF_NEG, BPF_MOD, BPF_XOR, BPF_JA, BPF_JEQ,
        BPF_JGT, BPF_JGE, BPF_JSET, BPF_K, BPF_X, BPF_CLASS, BPF_SIZE, BPF_MODE,
        BPF_OP, BPF_SRC, BPF_STMT, BPF_JUMP, BPF_PROG, BPF_FILTER,
        # seccomp macros
        SECCOMP_MODE_DISABLED, SECCOMP_MODE_STRICT, SECCOMP_MODE_FILTER,
        SECCOMP_SET_MODE_STRICT, SECCOMP_SET_MODE_FILTER,
        SECCOMP_GET_ACTION_AVAIL, SECCOMP_GET_NOTIF_SIZES,
        SECCOMP_FILTER_FLAG_TSYNC, SECCOMP_FILTER_FLAG_LOG,
        SECCOMP_FILTER_FLAG_SPEC_ALLOW, SECCOMP_FILTER_FLAG_NEW_LISTENER,
        SECCOMP_RET_KILL_PROCESS, SECCOMP_RET_KILL_THREAD, SECCOMP_RET_KILL,
        SECCOMP_RET_TRAP, SECCOMP_RET_ERRNO, SECCOMP_RET_USER_NOTIF,
        SECCOMP_RET_TRACE, SECCOMP_RET_LOG, SECCOMP_RET_ALLOW,
        SECCOMP_RET_ACTION_FULL, SECCOMP_RET_ACTION, SECCOMP_RET_DATA)