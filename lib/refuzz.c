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

#define COVERAGE_SIZE_TAG "/refuzz_size"

static uint8_t *edge_cnt;
static size_t edge_sz;

void __sanitizer_cov_trace_pc_guard_init(uint32_t *start, uint32_t *stop) {
    const char *name = getenv("REFUZZ_COVERAGE");
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

    const char *szname = COVERAGE_SIZE_TAG;
    fd = shm_open(szname, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd == -1) return;
    if (ftruncate(fd, sizeof(uint32_t)) == -1) return;
    uint32_t *sz = mmap(NULL, edge_sz, PROT_WRITE, MAP_SHARED, fd, 0);
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
