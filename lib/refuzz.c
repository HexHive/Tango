// Compile binaries with -fsanitize-coverage={func, bb, edge},trace-pc-guard

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sanitizer/coverage_interface.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>

#define COVERAGE_SIZE_TAG "/refuzz_size"

static uint8_t *edge_cnt;
static size_t edge_sz;

void __sanitizer_cov_trace_pc_guard_init(uint32_t *start, uint32_t *stop) {
    const char *name = getenv("REFUZZ_COVERAGE");
    if (!name) {
        fputs("env:REFUZZ_COVERAGE not specified, disabling instrumentation\n", stderr);
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
    fd = shm_open(name, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd == -1) return;
    if (ftruncate(fd, sizeof(uint32_t)) == -1) return;
    uint32_t *sz = mmap(NULL, edge_sz, PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (!sz) return;
    *sz = edge_sz;
    msync(sz, sizeof(uint32_t), MS_SYNC | MS_INVALIDATE);
    munmap(sz, sizeof(uint32_t));
}

void __sanitizer_cov_trace_pc_guard(uint32_t *guard) {
    if (!*guard) return;

    if (!__builtin_add_overflow(edge_cnt[*guard], 1, &edge_cnt[*guard]))
        edge_cnt[*guard] = UINT8_MAX;

    msync(edge_cnt, edge_sz, MS_ASYNC | MS_INVALIDATE);
}