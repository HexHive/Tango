from tango.ptrace.cpu_info import (
    CPU_64BITS, CPU_PPC32, CPU_PPC64, CPU_ARM32, CPU_AARCH64)
from ctypes import (Structure, Union, sizeof, POINTER,
                    c_char, c_ushort, c_int, c_uint, c_ulong, c_void_p, c_int32,
                    c_uint8, c_uint16, c_uint32, c_uint64, c_size_t, c_int64)
import signal

pid_t = c_int
uid_t = c_ushort
clock_t = c_uint

# From /usr/include/asm-i386/user.h
# Also more reliably in the kernel sources:
# arch/$ARCH/include/uapi/asm/ptrace.h


class register_structure(Structure):
    def __str__(self):
        regs = {}
        for reg in self.__class__._fields_:
            regs.update({reg[0]: getattr(self, reg[0])})
        return str(regs)


class user_regs_struct(register_structure):
    if CPU_PPC32:
        _fields_ = (
            ("gpr0", c_ulong),
            ("gpr1", c_ulong),
            ("gpr2", c_ulong),
            ("gpr3", c_ulong),
            ("gpr4", c_ulong),
            ("gpr5", c_ulong),
            ("gpr6", c_ulong),
            ("gpr7", c_ulong),
            ("gpr8", c_ulong),
            ("gpr9", c_ulong),
            ("gpr10", c_ulong),
            ("gpr11", c_ulong),
            ("gpr12", c_ulong),
            ("gpr13", c_ulong),
            ("gpr14", c_ulong),
            ("gpr15", c_ulong),
            ("gpr16", c_ulong),
            ("gpr17", c_ulong),
            ("gpr18", c_ulong),
            ("gpr19", c_ulong),
            ("gpr20", c_ulong),
            ("gpr21", c_ulong),
            ("gpr22", c_ulong),
            ("gpr23", c_ulong),
            ("gpr24", c_ulong),
            ("gpr25", c_ulong),
            ("gpr26", c_ulong),
            ("gpr27", c_ulong),
            ("gpr28", c_ulong),
            ("gpr29", c_ulong),
            ("gpr30", c_ulong),
            ("gpr31", c_ulong),
            ("nip", c_ulong),
            ("msr", c_ulong),
            ("orig_gpr3", c_ulong),
            ("ctr", c_ulong),
            ("link", c_ulong),
            ("xer", c_ulong),
            ("ccr", c_ulong),
            ("mq", c_ulong),  # FIXME: ppc64 => softe
            ("trap", c_ulong),
            ("dar", c_ulong),
            ("dsisr", c_ulong),
            ("result", c_ulong),
        )
    elif CPU_PPC64:
        _fields_ = (
            ("gpr0", c_ulong),
            ("gpr1", c_ulong),
            ("gpr2", c_ulong),
            ("gpr3", c_ulong),
            ("gpr4", c_ulong),
            ("gpr5", c_ulong),
            ("gpr6", c_ulong),
            ("gpr7", c_ulong),
            ("gpr8", c_ulong),
            ("gpr9", c_ulong),
            ("gpr10", c_ulong),
            ("gpr11", c_ulong),
            ("gpr12", c_ulong),
            ("gpr13", c_ulong),
            ("gpr14", c_ulong),
            ("gpr15", c_ulong),
            ("gpr16", c_ulong),
            ("gpr17", c_ulong),
            ("gpr18", c_ulong),
            ("gpr19", c_ulong),
            ("gpr20", c_ulong),
            ("gpr21", c_ulong),
            ("gpr22", c_ulong),
            ("gpr23", c_ulong),
            ("gpr24", c_ulong),
            ("gpr25", c_ulong),
            ("gpr26", c_ulong),
            ("gpr27", c_ulong),
            ("gpr28", c_ulong),
            ("gpr29", c_ulong),
            ("gpr30", c_ulong),
            ("gpr31", c_ulong),
            ("nip", c_ulong),
            ("msr", c_ulong),
            ("orig_gpr3", c_ulong),
            ("ctr", c_ulong),
            ("link", c_ulong),
            ("xer", c_ulong),
            ("ccr", c_ulong),
            ("softe", c_ulong),
            ("trap", c_ulong),
            ("dar", c_ulong),
            ("dsisr", c_ulong),
            ("result", c_ulong),
        )
    elif CPU_ARM32:
        _fields_ = tuple(("r%i" % reg, c_ulong) for reg in range(18))
    elif CPU_AARCH64:
        _fields_ = tuple([*[("r%i" % reg, c_ulong) for reg in range(31)],
                         ('sp', c_ulong),
                         ('pc', c_ulong),
                         ('pstate', c_ulong)]
                         )
    elif CPU_64BITS:
        _fields_ = (
            ("r15", c_ulong),
            ("r14", c_ulong),
            ("r13", c_ulong),
            ("r12", c_ulong),
            ("rbp", c_ulong),
            ("rbx", c_ulong),
            ("r11", c_ulong),
            ("r10", c_ulong),
            ("r9", c_ulong),
            ("r8", c_ulong),
            ("rax", c_ulong),
            ("rcx", c_ulong),
            ("rdx", c_ulong),
            ("rsi", c_ulong),
            ("rdi", c_ulong),
            ("orig_rax", c_ulong),
            ("rip", c_ulong),
            ("cs", c_ulong),
            ("eflags", c_ulong),
            ("rsp", c_ulong),
            ("ss", c_ulong),
            ("fs_base", c_ulong),
            ("gs_base", c_ulong),
            ("ds", c_ulong),
            ("es", c_ulong),
            ("fs", c_ulong),
            ("gs", c_ulong)
        )
    else:
        _fields_ = (
            ("ebx", c_ulong),
            ("ecx", c_ulong),
            ("edx", c_ulong),
            ("esi", c_ulong),
            ("edi", c_ulong),
            ("ebp", c_ulong),
            ("eax", c_ulong),
            ("ds", c_ushort),
            ("__ds", c_ushort),
            ("es", c_ushort),
            ("__es", c_ushort),
            ("fs", c_ushort),
            ("__fs", c_ushort),
            ("gs", c_ushort),
            ("__gs", c_ushort),
            ("orig_eax", c_ulong),
            ("eip", c_ulong),
            ("cs", c_ushort),
            ("__cs", c_ushort),
            ("eflags", c_ulong),
            ("esp", c_ulong),
            ("ss", c_ushort),
            ("__ss", c_ushort),
        )


class user_fpregs_struct(register_structure):
    if CPU_64BITS:
        _fields_ = (
            ("cwd", c_uint16),
            ("swd", c_uint16),
            ("ftw", c_uint16),
            ("fop", c_uint16),
            ("rip", c_uint64),
            ("rdp", c_uint64),
            ("mxcsr", c_uint32),
            ("mxcr_mask", c_uint32),
            ("st_space", c_uint32 * 32),
            ("xmm_space", c_uint32 * 64),
            ("padding", c_uint32 * 24)
        )
    else:
        _fields_ = (
            ("cwd", c_ulong),
            ("swd", c_ulong),
            ("twd", c_ulong),
            ("fip", c_ulong),
            ("fcs", c_ulong),
            ("foo", c_ulong),
            ("fos", c_ulong),
            ("st_space", c_ulong * 20)
        )


if not CPU_64BITS:
    class user_fpxregs_struct(register_structure):
        _fields_ = (
            ("cwd", c_ushort),
            ("swd", c_ushort),
            ("twd", c_ushort),
            ("fop", c_ushort),
            ("fip", c_ulong),
            ("fcs", c_ulong),
            ("foo", c_ulong),
            ("fos", c_ulong),
            ("mxcsr", c_ulong),
            ("reserved", c_ulong),
            ("st_space", c_ulong * 32),
            ("xmm_space", c_ulong * 32),
            ("padding", c_ulong * 56)
        )

# From /usr/include/asm-generic/siginfo.h


class _sifields_sigfault_t(Union):
    _fields_ = (
        ("_addr", c_void_p),
    )


class _sifields_sigchld_t(Structure):
    _fields_ = (
        ("pid", pid_t),
        ("uid", uid_t),
        ("status", c_int),
        ("utime", clock_t),
        ("stime", clock_t),
    )


class _sifields_t(Union):
    _fields_ = (
        ("pad", c_char * (128 - 3 * sizeof(c_int))),
        ("_sigchld", _sifields_sigchld_t),
        ("_sigfault", _sifields_sigfault_t),
        #        ("_kill", _sifields_kill_t),
        #        ("_timer", _sifields_timer_t),
        #        ("_rt", _sifields_rt_t),
        #        ("_sigpoll", _sifields_sigpoll_t),
    )


class siginfo(Structure):
    _fields_ = (
        ("si_signo", c_int),
        ("si_errno", c_int),
        ("si_code", c_int),
        ("_sifields", _sifields_t)
    )
    _anonymous_ = ("_sifields",)


class iovec_struct(Structure):
    _fields_ = (
        ("buf", c_void_p),
        ("len", c_size_t)
    )

class syscall_info_entry(Structure):
    _fields_ = (
        ("nr", c_uint64),
        ("args", c_uint64*6),
    )

class syscall_info_exit(Structure):
    _fields_ = (
        ("rval", c_int64),
        ("is_error", c_uint8),
    )

class syscall_info_seccomp(Structure):
    _fields_ = (
        ("nr", c_uint64),
        ("args", c_uint64*6),
        ("ret_data", c_uint32),
    )

class syscall_info_data(Union):
    _fields_ = (
        ("entry", syscall_info_entry),
        ("exit", syscall_info_exit),
        ("seccomp", syscall_info_seccomp),
    )

class ptrace_syscall_info(Structure):
    _fields_ = (
        ("op", c_uint8),
        ("arch", c_uint32),
        ("instruction_pointer", c_void_p),
        ("stack_pointer", c_void_p),
        ("_data", syscall_info_data),
    )
    _anonymous_ = ("_data",)

class sock_filter(Structure):
    _fields_ = (
        ("code", c_uint16),
        ("jt", c_uint8),
        ("jf", c_uint8),
        ("k", c_uint32),
    )

class sock_fprog(Structure):
    _fields_ = (
        ("len", c_uint16),
        ("filter", POINTER(sock_filter)),
    )

class seccomp_data(Structure):
    _fields_ = (
        ("nr", c_uint32),
        ("arch", c_uint32),
        ("instruction_pointer", c_uint64),
        ("args", c_uint64*6),
    )

class signalfd_siginfo(Structure):
    _fields_ = (
        ("ssi_signo", c_uint32),     # Signal number
        ("ssi_errno", c_int32),      # Error number (unused)
        ("ssi_code", c_int32),       # Signal code
        ("ssi_pid", c_uint32),       # PID of sender
        ("ssi_uid", c_uint32),       # Real UID of sender
        ("ssi_fd", c_int32),         # File descriptor (SIGIO)
        ("ssi_tid", c_uint32),       # Kernel timer ID (POSIX timers)
        ("ssi_band", c_uint32),      # Band event (SIGIO)
        ("ssi_overrun", c_uint32),   # POSIX timer overrun count
        ("ssi_trapno", c_uint32),    # Trap number that caused signal
        ("ssi_status", c_int32),     # Exit status or signal (SIGCHLD)
        ("ssi_int", c_int32),        # Integer sent by sigqueue(3)
        ("ssi_ptr", c_uint64),       # Pointer sent by sigqueue(3)
        ("ssi_utime", c_uint64),     # User CPU time consumed (SIGCHLD)
        ("ssi_stime", c_uint64),     # System CPU time consumed
                                     #   (SIGCHLD)
        ("ssi_addr", c_uint64),      # Address that generated signal
                                     #   (for hardware-generated signals)
        ("ssi_addr_lsb", c_uint16),  # Least significant bit of address
                                     #   (SIGBUS; since Linux 2.6.37)
        ("pad", c_uint8*46),         # Pad size to 128 bytes (allow for
                                     #   additional fields in the future)
    )

class _sigset(Structure):
    _fields_ = []
    _signals = sorted(signal.valid_signals())
    _i = _j = 0
    while _i < signal.Signals.SIGRTMAX:
        _i += 1
        if int(s := _signals[_j]) == _i:
            _j += 1
            signum = int(s)
            if type(s) is int:
                signame = f'SIG{signum:03}'
            else:
                signame = s.name
            bitfield = (signame, c_uint64, 1)
            _fields_.append(bitfield)
        else:
            _fields_.append(('_pad', c_uint64, 1))
    del _i, _j, _signals

class sigset(Union):
    _fields_ = (
        ('_sigs', _sigset),
        ('mask', c_uint64)
    )
    _anonymous_ = '_sigs',
