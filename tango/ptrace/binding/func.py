from tango.ptrace import PtraceError
from tango.ptrace.ctypes_tools import formatAddress
from tango.ptrace.os_tools import (RUNNING_LINUX, RUNNING_BSD, RUNNING_OPENBSD,
    HAS_PROC)
from tango.ptrace.cpu_info import CPU_64BITS, CPU_WORD_SIZE, CPU_POWERPC, CPU_AARCH64

from os import strerror
from functools import reduce
from ctypes import (addressof, get_errno, set_errno, sizeof, Structure,
    Array, POINTER, byref)
from ctypes import (
    c_int, c_void_p, c_size_t, c_ssize_t, c_char_p, c_ulong
)

if RUNNING_OPENBSD:
    from tango.ptrace.binding.openbsd_struct import (
        reg as ptrace_registers_t,
        fpreg as user_fpregs_struct)

elif RUNNING_BSD:
    from tango.ptrace.binding.freebsd_struct import (
        reg as ptrace_registers_t)

elif RUNNING_LINUX:
    from tango.ptrace.binding.linux_struct import (
        user_regs_struct as ptrace_registers_t,
        user_fpregs_struct, siginfo, iovec_struct, ptrace_syscall_info,
        sock_filter, sock_fprog)
    if not CPU_64BITS:
        from tango.ptrace.binding.linux_struct import user_fpxregs_struct
else:
    raise NotImplementedError("Unknown OS!")
REGISTER_NAMES = tuple(name for name, type in ptrace_registers_t._fields_)

HAS_PTRACE_SINGLESTEP = True
HAS_PTRACE_EVENTS = False
HAS_PTRACE_IO = False
HAS_PTRACE_SIGINFO = False
HAS_PTRACE_GETREGS = False
HAS_PTRACE_GETREGSET = False
HAS_PTRACE_SETREGS = False
HAS_PTRACE_SETREGSET = False
HAS_SECCOMP_FILTER = False

# Special flags that are required to wait for cloned processes (threads)
# See wait(2)
THREAD_TRACE_FLAGS = 0x00000000

pid_t = c_int

# PTRACE_xxx constants from /usr/include/sys/ptrace.h
# (Linux 2.6.21 Ubuntu Feisty i386)
PTRACE_TRACEME = 0
PTRACE_PEEKTEXT = 1
PTRACE_PEEKDATA = 2
PTRACE_PEEKUSER = 3
PTRACE_POKETEXT = 4
PTRACE_POKEDATA = 5
PTRACE_POKEUSER = 6
PTRACE_CONT = 7
PTRACE_KILL = 8
if HAS_PTRACE_SINGLESTEP:
    PTRACE_SINGLESTEP = 9

if RUNNING_OPENBSD:
    # OpenBSD 4.2 i386
    PTRACE_ATTACH = 9
    PTRACE_DETACH = 10
    HAS_PTRACE_GETREGS = True
    PTRACE_GETREGS = 33
    PTRACE_SETREGS = 34
    PTRACE_GETFPREGS = 35
    PTRACE_SETFPREGS = 36
    HAS_PTRACE_IO = True
    PTRACE_IO = 11
    HAS_PTRACE_SINGLESTEP = True
    PTRACE_SINGLESTEP = 32  # PT_STEP
    # HAS_PTRACE_EVENTS = True
    # PTRACE_SETOPTIONS = 12 # PT_SET_EVENT_MASK
    # PTRACE_GETEVENTMSG = 14 # PT_GET_PROCESS_STATE
elif RUNNING_BSD:
    # FreeBSD 7.0RC1 i386
    PTRACE_ATTACH = 10
    PTRACE_DETACH = 11
    PTRACE_SYSCALL = 22
    if not CPU_POWERPC:
        HAS_PTRACE_GETREGS = True
        PTRACE_GETREGS = 33
    PTRACE_SETREGS = 34
    HAS_PTRACE_IO = True
    PTRACE_IO = 12
else:
    # Linux
    if not CPU_AARCH64:
        HAS_PTRACE_GETREGS = True
        HAS_PTRACE_SETREGS = True
        PTRACE_GETREGS = 12
        PTRACE_SETREGS = 13

    HAS_PTRACE_GETREGSET = True
    HAS_PTRACE_SETREGSET = True
    PTRACE_GETREGSET = 0x4204
    PTRACE_SETREGSET = 0x4205
    NT_PRSTATUS = 1

    PTRACE_ATTACH = 16
    PTRACE_DETACH = 17
    PTRACE_SYSCALL = 24
if RUNNING_LINUX:
    PTRACE_GETFPREGS = 14
    PTRACE_SETFPREGS = 15
    if not CPU_64BITS:
        PTRACE_GETFPXREGS = 18
        PTRACE_SETFPXREGS = 19
    HAS_PTRACE_SIGINFO = True
    PTRACE_GETSIGINFO = 0x4202
    PTRACE_SETSIGINFO = 0x4203

    HAS_PTRACE_EVENTS = True
    PTRACE_SETOPTIONS = 0x4200
    PTRACE_GETEVENTMSG = 0x4201
    PTRACE_GET_SYSCALL_INFO = 0x420e

    # Linux introduces the __WALL flag for wait
    # Revisit: __WALL is the default for ptraced children since Linux 4.7
    THREAD_TRACE_FLAGS = 0x40000000

    ## BPF constants
    # Instruction classes
    BPF_LD = 0x00
    BPF_LDX = 0x01
    BPF_ST = 0x02
    BPF_STX = 0x03
    BPF_ALU = 0x04
    BPF_JMP = 0x05
    BPF_RET = 0x06
    BPF_MISC = 0x07

    # ld/ldx fields
    BPF_W = 0x00 # 32-bit
    BPF_H = 0x08 # 16-bit
    BPF_B = 0x10 #  8-bit
    BPF_IMM = 0x00
    BPF_ABS = 0x20
    BPF_IND = 0x40
    BPF_MEM = 0x60
    BPF_LEN = 0x80
    BPF_MSH = 0xa0

    # alu/jmp fields
    BPF_ADD = 0x00
    BPF_SUB = 0x10
    BPF_MUL = 0x20
    BPF_DIV = 0x30
    BPF_OR = 0x40
    BPF_AND = 0x50
    BPF_LSH = 0x60
    BPF_RSH = 0x70
    BPF_NEG = 0x80
    BPF_MOD = 0x90
    BPF_XOR = 0xa0

    BPF_JA = 0x00
    BPF_JEQ = 0x10
    BPF_JGT = 0x20
    BPF_JGE = 0x30
    BPF_JSET = 0x40
    BPF_K = 0x00
    BPF_X = 0x08

    ## BPF Macros
    BPF_CLASS = lambda code: ((code) & 0x07)
    BPF_SIZE = lambda code: ((code) & 0x18)
    BPF_MODE = lambda code: ((code) & 0xe0)
    BPF_OP = lambda code: ((code) & 0xf0)
    BPF_SRC = lambda code: ((code) & 0x08)
    BPF_STMT = lambda code, k: sock_filter(code, 0, 0, k)
    BPF_JUMP = lambda code, k, jt, jf: sock_filter(code, jt, jf, k)
    BPF_PROG = lambda filt: sock_fprog(len(filt), filt)

    def BPF_FILTER(*ops):
        class sock_filter_array(Array):
            _length_ = len(ops)
            _type_ = sock_filter
        return sock_filter_array(*ops)

if HAS_PROC:
    from tango.ptrace.linux_proc import readProc, ProcError
    config = None
    try:
        configz = readProc('config.gz', mode='rb')
    except ProcError:
        import os, platform
        path = f'/boot/config-{platform.release()}'
        try:
            with open(path, 'rt') as f:
                config = f.read()
        except IOError:
            pass
    else:
        import zlib
        config = zlib.decompress(configz)
    if config:
        if 'CONFIG_SECCOMP_FILTER=y' in config:
            HAS_SECCOMP_FILTER = True

            # Valid values for seccomp.mode and prctl(PR_SET_SECCOMP, <mode>)
            SECCOMP_MODE_DISABLED = 0 # seccomp is not in use.
            SECCOMP_MODE_STRICT = 1 # uses hard-coded filter.
            SECCOMP_MODE_FILTER = 2 # uses user-supplied filter.

            # Valid operations for seccomp syscall.
            SECCOMP_SET_MODE_STRICT = 0
            SECCOMP_SET_MODE_FILTER = 1
            SECCOMP_GET_ACTION_AVAIL = 2
            SECCOMP_GET_NOTIF_SIZES = 3

            # Valid flags for SECCOMP_SET_MODE_FILTER
            SECCOMP_FILTER_FLAG_TSYNC = 1 << 0
            SECCOMP_FILTER_FLAG_LOG = 1 << 1
            SECCOMP_FILTER_FLAG_SPEC_ALLOW = 1 << 2
            SECCOMP_FILTER_FLAG_NEW_LISTENER = 1 << 3

            # Return values
            SECCOMP_RET_KILL_PROCESS = 0x80000000 # kill the process
            SECCOMP_RET_KILL_THREAD = 0x00000000 # kill the thread
            SECCOMP_RET_KILL = SECCOMP_RET_KILL_THREAD
            SECCOMP_RET_TRAP = 0x00030000 # disallow and force a SIGSYS
            SECCOMP_RET_ERRNO = 0x00050000 # returns an errno
            SECCOMP_RET_USER_NOTIF = 0x7fc00000 # notifies userspace
            SECCOMP_RET_TRACE = 0x7ff00000 # pass to a tracer or disallow
            SECCOMP_RET_LOG = 0x7ffc0000 # allow after logging
            SECCOMP_RET_ALLOW = 0x7fff0000 # allow

            # Masks for the return value sections.
            SECCOMP_RET_ACTION_FULL = 0xffff0000
            SECCOMP_RET_ACTION = 0x7fff0000
            SECCOMP_RET_DATA = 0x0000ffff


PTRACE_O_TRACESYSGOOD = 0x00000001
PTRACE_O_TRACEFORK = 0x00000002
PTRACE_O_TRACEVFORK = 0x00000004
PTRACE_O_TRACECLONE = 0x00000008
PTRACE_O_TRACEEXEC = 0x00000010
PTRACE_O_TRACEVFORKDONE = 0x00000020
PTRACE_O_TRACEEXIT = 0x00000040
PTRACE_O_TRACESECCOMP = 0x00000080

# Wait extended result codes for the above trace options
PTRACE_EVENT_FORK = 1
PTRACE_EVENT_VFORK = 2
PTRACE_EVENT_CLONE = 3
PTRACE_EVENT_EXEC = 4
PTRACE_EVENT_VFORK_DONE = 5
PTRACE_EVENT_EXIT = 6
PTRACE_EVENT_SECCOMP = 7

try:
    from cptrace import ptrace as _ptrace
    HAS_CPTRACE = True
except ImportError:
    HAS_CPTRACE = False
    from ctypes import c_long, c_ulong
    from tango.ptrace.ctypes_libc import libc

    # Load ptrace() function from the system C library
    _ptrace = libc.ptrace
    _ptrace.argtypes = (c_ulong, c_ulong, c_ulong, c_ulong)
    _ptrace.restype = c_ulong

def ptrace(command, pid=0, arg1=0, arg2=0, check_errno=False):
    if HAS_CPTRACE:
        set_errno(0)
        result = _ptrace(command, pid, arg1, arg2, check_errno)
    else:
        result = _ptrace(command, pid, arg1, arg2)
        result_signed = c_long(result).value
        if result_signed == -1:
            errno = get_errno()
            # peek operations may returns -1 with errno=0:
            # it's not an error. For other operations, -1
            # is always an error
            if not(check_errno) or errno:
                message = "ptrace(cmd=%s, pid=%s, %r, %r) error #%s: %s" % (
                    command, pid, arg1, arg2,
                    errno, strerror(errno))
                raise PtraceError(message, errno=errno, pid=pid)
    return result


def ptrace_traceme():
    ptrace(PTRACE_TRACEME)


def ptrace_attach(pid):
    ptrace(PTRACE_ATTACH, pid)


def ptrace_detach(pid, signal=0):
    ptrace(PTRACE_DETACH, pid, 0, signal)


def _peek(command, pid, address):
    if address % CPU_WORD_SIZE:
        raise PtraceError(
            "ptrace can't read a word from an unaligned address (%s)!"
            % formatAddress(address), pid=pid)
    return ptrace(command, pid, address, check_errno=True)


def _poke(command, pid, address, word):
    if address % CPU_WORD_SIZE:
        raise PtraceError(
            "ptrace can't write a word to an unaligned address (%s)!"
            % formatAddress(address), pid=pid)
    ptrace(command, pid, address, word)


def ptrace_peektext(pid, address):
    return _peek(PTRACE_PEEKTEXT, pid, address)


def ptrace_peekdata(pid, address):
    return _peek(PTRACE_PEEKDATA, pid, address)


def ptrace_peekuser(pid, address):
    return _peek(PTRACE_PEEKUSER, pid, address)


def ptrace_poketext(pid, address, word):
    _poke(PTRACE_POKETEXT, pid, address, word)


def ptrace_pokedata(pid, address, word):
    _poke(PTRACE_POKEDATA, pid, address, word)


def ptrace_pokeuser(pid, address, word):
    _poke(PTRACE_POKEUSER, pid, address, word)


def ptrace_kill(pid):
    ptrace(PTRACE_KILL, pid)


if HAS_PTRACE_EVENTS:
    def WPTRACEEVENT(status):
        return status >> 16

    def ptrace_setoptions(pid, options):
        ptrace(PTRACE_SETOPTIONS, pid, 0, options)

    def ptrace_geteventmsg(pid):
        new_pid = pid_t()
        ptrace(PTRACE_GETEVENTMSG, pid, 0, addressof(new_pid))
        return new_pid.value

if RUNNING_LINUX:
    def ptrace_syscall(pid, signum=0):
        ptrace(PTRACE_SYSCALL, pid, 0, signum)

    def ptrace_cont(pid, signum=0):
        ptrace(PTRACE_CONT, pid, 0, signum)

    def ptrace_getsiginfo(pid):
        info = siginfo()
        ptrace(PTRACE_GETSIGINFO, pid, 0, addressof(info))
        return info

    def ptrace_setsiginfo(pid, info):
        ptrace(PTRACE_SETSIGINFO, pid, 0, addressof(info))

    def ptrace_getfpregs(pid):
        fpregs = user_fpregs_struct()
        ptrace(PTRACE_GETFPREGS, pid, 0, addressof(fpregs))
        return fpregs

    def ptrace_setfpregs(pid, fpregs):
        ptrace(PTRACE_SETFPREGS, pid, 0, addressof(fpregs))

    def ptrace_get_syscall_info(pid):
        info = ptrace_syscall_info()
        size = ptrace(PTRACE_GET_SYSCALL_INFO, pid, sizeof(ptrace_syscall_info),
            addressof(info))
        return info


    if not CPU_64BITS:
        def ptrace_getfpxregs(pid):
            fpxregs = user_fpxregs_struct()
            ptrace(PTRACE_GETFPXREGS, pid, 0, addressof(fpxregs))
            return fpxregs

        def ptrace_setfpxregs(pid, fpxregs):
            ptrace(PTRACE_SETFPXREGS, pid, 0, addressof(fpxregs))

    if HAS_PTRACE_GETREGS:
        def ptrace_getregs(pid):
            regs = ptrace_registers_t()
            ptrace(PTRACE_GETREGS, pid, 0, addressof(regs))
            return regs

    elif HAS_PTRACE_GETREGSET:
        def ptrace_getregs(pid):
            regs = ptrace_registers_t()
            iov = iovec_struct()
            setattr(iov, "buf", addressof(regs))
            setattr(iov, "len", sizeof(regs))
            ptrace(PTRACE_GETREGSET, pid, NT_PRSTATUS, addressof(iov))
            return regs

    if HAS_PTRACE_SETREGS:
        def ptrace_setregs(pid, regs):
            ptrace(PTRACE_SETREGS, pid, 0, addressof(regs))

    elif HAS_PTRACE_SETREGSET:
        def ptrace_setregs(pid, regs):
            iov = iovec_struct()
            setattr(iov, "buf", addressof(regs))
            setattr(iov, "len", sizeof(regs))
            ptrace(PTRACE_SETREGSET, pid, NT_PRSTATUS, addressof(iov))

    if HAS_PTRACE_SINGLESTEP:
        def ptrace_singlestep(pid):
            ptrace(PTRACE_SINGLESTEP, pid)

else:
    def ptrace_syscall(pid, signum=0):
        ptrace(PTRACE_SYSCALL, pid, 1, signum)

    def ptrace_cont(pid, signum=0):
        ptrace(PTRACE_CONT, pid, 1, signum)

    if HAS_PTRACE_GETREGS:
        def ptrace_getregs(pid):
            regs = ptrace_registers_t()
            ptrace(PTRACE_GETREGS, pid, addressof(regs))
            return regs

    def ptrace_setregs(pid, regs):
        ptrace(PTRACE_SETREGS, pid, addressof(regs))

    if HAS_PTRACE_SINGLESTEP:
        def ptrace_singlestep(pid):
            ptrace(PTRACE_SINGLESTEP, pid, 1)

if HAS_PTRACE_IO:
    def ptrace_io(pid, io_desc):
        ptrace(PTRACE_IO, pid, addressof(io_desc))

if RUNNING_LINUX:
    HAS_SIGNALFD = True
    from tango.ptrace.binding.linux_struct import sigset, signalfd_siginfo
    from tango.ptrace.ctypes_libc import libc
    from signal import Signals

    prctl = libc.prctl
    prctl.restype = c_int
    prctl.argtypes = (c_int, c_ulong, c_ulong, c_ulong, c_ulong)

    PR_SET_PDEATHSIG = 1
    PR_CAPBSET_READ = 23
    PR_CAPBSET_DROP = 24
    PR_SET_NO_NEW_PRIVS = 38

    signalfd = libc.signalfd
    signalfd.argtypes = (
        c_int,            # fd
        POINTER(sigset),  # mask
        c_int,            # flags
    )
    signalfd.restype = c_int

    read = libc.read
    read.argtypes = (
        c_int,            # fd
        c_void_p,         # buf
        c_size_t,         # size
    )
    read.restype = c_ssize_t

    SFD_CLOEXEC  = 0o02000000
    SFD_NONBLOCK = 0o00004000

    def create_signalfd(*sigs, flags=0):
        sset = sigset()
        for sig in sigs:
            if isinstance(sig, str):
                if not hasattr(sset, sig):
                    raise ValueError(f"{sig} is an invalid signal name.")
                setattr(sset, sig, 1)
            elif isinstance(sig, Signals):
                sset.mask |= 1 << (sig.value - 1)
            elif isinstance(sig, int) and sig < Signals.SIGRTMAX:
                sset.mask |= 1 << (sig - 1)
            else:
                raise ValueError(f"{sig} is not a valid signal identifier")
        fd = signalfd(-1, byref(sset), flags)
        if fd < 0:
            error = strerror(get_errno())
            raise RuntimeError(f"Failed to create signalfd: {error}")
        return fd

    def read_signalfd(fd, sinfo=None):
        sinfo = sinfo or signalfd_siginfo()
        rv = read(fd, byref(sinfo), sizeof(sinfo))
        if rv < 0:
            error = strerror(get_errno())
            raise BlockingIOError(f"Failed to read from signalfd {fd}: {error}")
        elif rv == 0:
            raise EOFError(f"End-of-file on signalfd {fd}")
        return sinfo

    HAS_NAMESPACES = True
    _mount = libc.mount
    _mount.restype = c_int
    _mount.argtypes = (
        c_char_p,  # source
        c_char_p,  # target
        c_char_p,  # filesystemtype
        c_ulong,   # mountflags
        c_void_p   # data
    )
    _umount = libc.umount2
    _umount.restype = c_int
    _umount.argtypes = (
        c_char_p,  # target
        c_int      # flags
    )

    MS_RDONLY =       1        # Mount read-only
    MS_NOSUID =       2        # Ignore suid and sgid bits
    MS_NODEV =        4        # Disallow access to device special files
    MS_NOEXEC =       8        # Disallow program execution
    MS_SYNCHRONOUS =  16       # Writes are synced at once
    MS_REMOUNT =      32       # Alter flags of a mounted FS
    MS_MANDLOCK =     64       # Allow mandatory locks on an FS
    MS_DIRSYNC =      128      # Directory modifications are synchronous
    MS_NOSYMFOLLOW =  256      # Do not follow symlinks
    MS_NOATIME =      1024     # Do not update access times.
    MS_NODIRATIME =   2048     # Do not update directory access times
    MS_BIND =         4096
    MS_MOVE =         8192
    MS_REC =          16384
    MS_SILENT =       32768
    MS_POSIXACL =     (1<<16)  # VFS does not apply the umask
    MS_UNBINDABLE =   (1<<17)  # change to unbindable
    MS_PRIVATE =      (1<<18)  # change to private
    MS_SLAVE =        (1<<19)  # change to slave
    MS_SHARED =       (1<<20)  # change to shared
    MS_RELATIME =     (1<<21)  # Update atime relative to mtime/ctime.
    MS_KERNMOUNT =    (1<<22)  # this is a kern_mount call
    MS_I_VERSION =    (1<<23)  # Update inode I_version field
    MS_STRICTATIME =  (1<<24)  # Always perform atime updates
    MS_LAZYTIME =     (1<<25)  # Update the on-disk [acm]times lazily

    ## Umount options
    MNT_FORCE =       0x00000001  # Attempt to forcibily umount
    MNT_DETACH =      0x00000002  # Just detach from the tree
    MNT_EXPIRE =      0x00000004  # Mark for expiry
    UMOUNT_NOFOLLOW = 0x00000008  # Don't follow symlink on umount
    UMOUNT_UNUSED =   0x80000000  # Flag guaranteed to be unused

    _unshare = libc.unshare
    _unshare.restype = c_int
    _unshare.argtypes = (c_int,)  # flags

    CLONE_FILES =     0x00000400  # set if open files shared between processes
    CLONE_FS =        0x00000200  # set if fs info shared between processes
    CLONE_NEWCGROUP = 0x02000000  # New cgroup namespace
    CLONE_NEWIPC =    0x08000000  # New ipc namespace
    CLONE_NEWNET =    0x40000000  # New network namespace
    CLONE_NEWNS =     0x00020000  # New mount namespace group
    CLONE_NEWPID =    0x20000000  # New pid namespace
    CLONE_NEWUSER =   0x10000000  # New user namespace
    CLONE_NEWUTS =    0x04000000  # New utsname namespace
    CLONE_SYSVSEM =   0x00040000  # share system V SEM_UNDO semantics

    _pivot_root = libc.pivot_root
    _pivot_root.restype = c_int
    _pivot_root.argtypes = (
        c_char_p,  # new_root
        c_char_p,  # old_root
    )

    CAP_CHOWN            = 0
    CAP_DAC_OVERRIDE     = 1
    CAP_DAC_READ_SEARCH  = 2
    CAP_FOWNER           = 3
    CAP_FSETID           = 4
    CAP_KILL             = 5
    CAP_SETGID           = 6
    CAP_SETUID           = 7
    CAP_SETPCAP          = 8
    CAP_LINUX_IMMUTABLE  = 9
    CAP_NET_BIND_SERVICE = 10
    CAP_NET_BROADCAST    = 11
    CAP_NET_ADMIN        = 12
    CAP_NET_RAW          = 13
    CAP_IPC_LOCK         = 14
    CAP_IPC_OWNER        = 15
    CAP_SYS_MODULE       = 16
    CAP_SYS_RAWIO        = 17
    CAP_SYS_CHROOT       = 18
    CAP_SYS_PTRACE       = 19
    CAP_SYS_PACCT        = 20
    CAP_SYS_ADMIN        = 21
    CAP_SYS_BOOT         = 22
    CAP_SYS_NICE         = 23
    CAP_SYS_RESOURCE     = 24
    CAP_SYS_TIME         = 25
    CAP_SYS_TTY_CONFIG   = 26
    CAP_MKNOD            = 27
    CAP_LEASE            = 28
    CAP_AUDIT_WRITE      = 29
    CAP_AUDIT_CONTROL    = 30
    CAP_SETFCAP          = 31
    CAP_MAC_OVERRIDE     = 32
    CAP_MAC_ADMIN        = 33
    CAP_SYSLOG           = 34
    CAP_WAKE_ALARM       = 35
    CAP_BLOCK_SUSPEND    = 36
    CAP_AUDIT_READ       = 37

    def _ensure_bytes(x, /):
        match x:
            case bytes(_):
                return x
            case str(_):
                return x.encode()
            case _:
                if not x:
                    return None
                else:
                    return str(x).encode()

    def mount(source, target, fstype, flags, data):
        source = _ensure_bytes(source)
        target = _ensure_bytes(target)
        fstype = _ensure_bytes(fstype)
        data = _ensure_bytes(data)
        if _mount(source, target, fstype, flags, data) < 0:
            error = strerror(get_errno())
            raise RuntimeError(error)

    def umount(target, flags=0):
        target = _ensure_bytes(target)
        if _umount(target, flags) < 0:
            error = strerror(get_errno())
            raise RuntimeError(error)

    def unshare(flags):
        if _unshare(flags) < 0:
            error = strerror(get_errno())
            raise RuntimeError(error)

    def pivot_root(new_root, old_root):
        new_root = _ensure_bytes(new_root)
        old_root = _ensure_bytes(old_root)
        if _pivot_root(new_root, old_root) < 0:
            error = strerror(get_errno())
            raise RuntimeError(error)

    def has_caps(*caps):
        for cap in caps:
            has_cap = bool(prctl(PR_CAPBSET_READ, cap, 0, 0, 0))
            if (errno := get_errno()):
                error = strerror(errno)
                raise RuntimeError(error)
            if not has_cap:
                return False
        return True

    def drop_caps(*caps):
        for cap in caps:
            rv = prctl(PR_CAPBSET_DROP, cap, 0, 0, 0)
            if rv < 0:
                error = strerror(get_errno())
                raise RuntimeError(error)

else:
    HAS_SIGNALFD = False
    HAS_NAMESPACES = False