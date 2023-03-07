// Compile binaries with -fsanitize-coverage={func, bb, edge},trace-pc-guard
// Minimum clang version: 13.0

#if !__has_feature(coverage_sanitizer)
#error Incompatible compiler! Please use Clang 13.0 or higher
#endif

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sanitizer/coverage_interface.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/prctl.h>
#include <linux/limits.h>

#ifdef USE_SECCOMP
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/syscall.h>
#endif

static uint8_t *edge_cnt;
static size_t edge_sz;

void __sanitizer_cov_trace_pc_guard_init(uint32_t *start, uint32_t *stop) {
    const char *name = getenv("TANGO_COVERAGE");
    const char *szname = getenv("TANGO_SIZE");
    if (!name) {
        for (uint32_t *x = start; x < stop; x++)
            *x = 0;  // disable all guards
        return;
    }

    static uint64_t N;  // Counter for the guards.
    if (start == stop || *start) return;  // Initialize only once.
    for (uint32_t *x = start; x < stop; x++)
        *x = ++N;  // Guards should start from 1.

    // initialize edge counters
    edge_sz = N * sizeof(uint8_t);
    int fd = shm_open(name, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd == -1) return;
    if (ftruncate(fd, edge_sz) == -1) return;
    edge_cnt = mmap(NULL, edge_sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (!edge_cnt) return;

    fd = shm_open(szname, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd == -1) return;
    if (ftruncate(fd, sizeof(uint32_t)) == -1) return;
    uint32_t *sz = mmap(NULL, sizeof(uint32_t), PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (!sz) return;
    *sz = edge_sz;
    munmap(sz, sizeof(uint32_t));
}

void __sanitizer_cov_trace_pc_guard(uint32_t *guard) {
    if (!*guard) return;

    if (__builtin_add_overflow(edge_cnt[*guard], 1, &edge_cnt[*guard]))
        edge_cnt[*guard] = UINT8_MAX;
}

int __wrap_bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen);
int __real_bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen);

pid_t __wrap_fork();
pid_t __real_fork();

long int __wrap_random(void);

__attribute__((no_sanitize("coverage")))
static inline void __reset_cov_map() {
    for (int i = 0; i < edge_sz; ++i)
        edge_cnt[i] = 0;
}

#ifdef USE_SECCOMP
__attribute__((no_sanitize("coverage")))
static inline void _install_seccomp_filter() {
    struct sock_filter filter[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, (offsetof(struct seccomp_data, nr))),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_accept, 16, 0),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_accept4, 15, 0),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_bind, 14, 0),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_close, 13, 0),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_dup, 12, 0),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_dup2, 11, 0),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_dup3, 10, 0),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_listen, 9, 0),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_poll, 8, 0),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_ppoll, 7, 0),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_read, 6, 0),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_recvfrom, 5, 0),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_recvmsg, 4, 0),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_select, 3, 0),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_shutdown, 2, 0),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_socket, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRACE | SECCOMP_RET_DATA),
    };
    struct sock_fprog prog = {
        .filter = filter,
        .len = (unsigned short) (sizeof(filter)/sizeof(filter[0])),
    };
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0))
        exit(-256);

    int flags = SECCOMP_FILTER_FLAG_TSYNC;
    if (syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, flags, &prog))
        exit(-256);
}
#endif

__attribute__((used, no_sanitize("coverage")))
static void _forkserver() {
    int fifofd = -1;
    const char *wd = getenv("TANGO_WORKDIR");
    char fifopath[PATH_MAX];
    snprintf(fifopath, PATH_MAX, "%s/%s", wd, "input.pipe");

#ifdef USE_SECCOMP
    _install_seccomp_filter();
#endif

    while(1) {
        __reset_cov_map();
        int child_pid = __real_fork();
        if (child_pid) {
            asm("int $3"); // trap and wait until fuzzer wakes us up
            int status, ret;
            //waitpid(child_pid, &status, 0);
            do {
                ret = waitpid(-1, &status, WNOHANG);
            } while (ret > 0);
        } else {
            fifofd = open(fifopath, O_RDONLY);
            dup2(fifofd, STDIN_FILENO);
            close(fifofd);
            break;
        }
    }
}

__attribute__((naked, used, no_sanitize("coverage")))
void forkserver() {
    asm volatile (
        "push %%rax\n"
        "push %%rcx\n"
        "push %%rdx\n"
        "push %%rsi\n"
        "push %%rdi\n"
        "push %%r8\n"
        "push %%r9\n"
        "push %%r10\n"
        "push %%r11\n"
        "call _forkserver\n"
        "pop %%r11\n"
        "pop %%r10\n"
        "pop %%r9\n"
        "pop %%r8\n"
        "pop %%rdi\n"
        "pop %%rsi\n"
        "pop %%rdx\n"
        "pop %%rcx\n"
        "pop %%rax\n"
        "ret"
        : /* No outputs */
        : /* No inputs */
        : /* No clobbers */
    );
}

static struct linger no_linger = {0};
static uint32_t reuse = 1;

/* Refer to this SO answer on the nitty-gritty about TIME_WAIT
 * https://stackoverflow.com/a/14388707
 */
__attribute__((no_sanitize("coverage")))
int __wrap_bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    setsockopt(sockfd, SOL_SOCKET, SO_LINGER, &no_linger, sizeof(no_linger));
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEPORT, &reuse, sizeof(reuse));
    printf("Enable address-port reuse on socket: %d\n", sockfd);

    /* go to bind then */
    return __real_bind(sockfd, addr, addrlen);
}

__attribute__((no_sanitize("coverage")))
pid_t __wrap_fork() {
    pid_t ppid = getpid();
    pid_t child_pid = __real_fork();
    if (child_pid == 0) {
        prctl(PR_SET_PDEATHSIG, SIGTERM);
        if (getppid() != ppid)
            exit(-256);
    }
    return child_pid;
}

__attribute__((no_sanitize("coverage")))
long int __wrap_random(void) {
    static long int x = 0;
    return ++x;
}