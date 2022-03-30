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

#define UNW_LOCAL_ONLY
#include "libunwind.h"

#define COVERAGE_SIZE_TAG "/refuzz_size"

static uint8_t *edge_cnt;
static size_t edge_sz;
static intptr_t *stack_hash;

/* STACK_DEPTH is defined in the Makefile */
static intptr_t stack_arr[STACK_DEPTH] = {0};
static size_t stack_idx = 0;

void __sanitizer_cov_trace_pc_guard_init(uint32_t *start, uint32_t *stop) {
    const char *cov_name = getenv("REFUZZ_COVERAGE");
    if (!cov_name) {
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
    int fd = shm_open(cov_name, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd == -1) return;
    if (ftruncate(fd, edge_sz) == -1) return;
    edge_cnt = mmap(NULL, edge_sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (!edge_cnt) return;

    const char *szname = COVERAGE_SIZE_TAG;
    fd = shm_open(szname, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd == -1) return;
    if (ftruncate(fd, sizeof(uint32_t)) == -1) return;
    uint32_t *sz = mmap(NULL, sizeof(uint32_t), PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (!sz) return;
    *sz = edge_sz;
    munmap(sz, sizeof(uint32_t));

    const char *stack_name = getenv("REFUZZ_STACK");
    if (!stack_name)
        return;
    fd = shm_open(stack_name, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd == -1) return;
    if (ftruncate(fd, sizeof(intptr_t)) == -1) return;
    stack_hash = mmap(NULL, sizeof(intptr_t), PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (!stack_hash) return;
    *stack_hash = 0;
}

void __sanitizer_cov_trace_pc_guard(uint32_t *guard) {
    if (!*guard) return;

    static int hits = 0;
    ++hits;
    printf("Hit count: %i\n", hits);

    if (__builtin_add_overflow(edge_cnt[*guard], 1, &edge_cnt[*guard]))
        edge_cnt[*guard] = UINT8_MAX;

    /* This may be excessive, but it seems to be the most straightforward place
     * for it. Use with -fsanitize-coverage=func
     */
    //stack_idx = (stack_idx + 1) % STACK_DEPTH;
    //stack_arr[stack_idx] = (intptr_t)__builtin_return_address(0);
    //*stack_hash = 0;
    //for (int i = 0; i < STACK_DEPTH; ++i) {
    //    *stack_hash ^= stack_arr[i];
    //}

    int i = 0;
    unw_cursor_t cursor; unw_context_t uc;
    unw_word_t ip;

    unw_getcontext(&uc);
    unw_init_local(&cursor, &uc);
    while (i < STACK_DEPTH && unw_step(&cursor) > 0) {
        unw_get_reg(&cursor, UNW_REG_IP, &ip);
        *stack_hash ^= (intptr_t)ip;
        ++i;
    }
}

__attribute__((no_sanitize("coverage")))
static inline void __reset_cov_map() {
    for (int i = 0; i < edge_sz; ++i)
        edge_cnt[i] = 0;
}

__attribute__((used, no_sanitize("coverage")))
static void _forkserver() {
    while(1) {
        __reset_cov_map();
        int child_pid = fork();
        if (child_pid) {
            asm("int $3"); // trap and wait until fuzzer wakes us up
            int status, ret;
            //waitpid(child_pid, &status, 0);
            do {
                ret = waitpid(-1, &status, WNOHANG);
            } while (ret > 0);
        } else {
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

int __wrap_bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen);
int __real_bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen);

pid_t __wrap_fork();
pid_t __real_fork();

/* Refer to this SO answer on the nitty-gritty about TIME_WAIT
 * https://stackoverflow.com/a/14388707
 */
int __wrap_bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    setsockopt(sockfd, SOL_SOCKET, SO_LINGER, &no_linger, sizeof(no_linger));
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEPORT, &reuse, sizeof(reuse));
    printf("Enable address-port reuse on socket: %d\n", sockfd);

    /* go to bind then */
    return __real_bind(sockfd, addr, addrlen);
}

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
