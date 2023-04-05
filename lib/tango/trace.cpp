#include "common.h"
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <linux/limits.h>

uint8_t *edge_cnt;
size_t edge_sz;

#ifdef __cplusplus
extern "C" {
#endif
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
    edge_cnt = (uint8_t *)mmap(
        NULL, edge_sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (!edge_cnt) return;

    fd = shm_open(szname, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd == -1) return;
    if (ftruncate(fd, sizeof(uint32_t)) == -1) return;
    uint32_t *sz = (uint32_t *)mmap(
        NULL, sizeof(uint32_t), PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (!sz) return;
    *sz = edge_sz;
    munmap(sz, sizeof(uint32_t));
}

void __sanitizer_cov_trace_pc_guard(uint32_t *guard) {
    if (!*guard) return;

    uint32_t idx = *guard - 1;
    if (__builtin_add_overflow(edge_cnt[idx], 1, &edge_cnt[idx]))
        edge_cnt[idx] = UINT8_MAX;
}
#ifdef __cplusplus
} // extern "C"
#endif