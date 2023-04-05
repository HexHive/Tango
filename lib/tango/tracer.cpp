#if !__has_feature(coverage_sanitizer)
#error Incompatible compiler! Please use Clang 13.0 or higher
#endif

#include "common.h"
#include "tracer.h"
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <linux/limits.h>
#include <stdexcept>
#include <assert.h>

namespace fuzzer {

Tracer CoverageTracer;

ATTRIBUTE_NO_SANITIZE_ALL
Tracer::Tracer() {
    initialized = false;
    disabled = false;
    num_guards = 0;
}

ATTRIBUTE_NO_SANITIZE_ALL
bool Tracer::InitializeMaps() {
    const char *covname = getenv("TANGO_COVERAGE");
    const char *szname = getenv("TANGO_SIZE");

    if (!covname) {
        disabled = true;
        return false;
    }

    try {
        // initialize edge counters
        size_t map_sz = num_guards * sizeof(uint8_t);
        int fd = shm_open(covname,
            O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
        if (fd == -1)
            throw std::runtime_error("Failed to open shm region");
        if (ftruncate(fd, map_sz) == -1)
            throw std::runtime_error("Failed to truncate shm region");
        feature_map = (uint8_t *)mmap(
            NULL, map_sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (!feature_map)
            throw std::runtime_error("Failed to mmap shm_region");

        fd = shm_open(szname,
            O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
        if (fd == -1)
            throw std::runtime_error("Failed to open shm region");
        if (ftruncate(fd, sizeof(uint32_t)) == -1)
            throw std::runtime_error("Failed to truncate shm region");
        uint32_t *sz = (uint32_t *)mmap(
            NULL, sizeof(uint32_t), PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (!sz)
            throw std::runtime_error("Failed to mmap shm_region");
        *sz = map_sz;
        munmap(sz, sizeof(uint32_t));
    } catch (...) {
        disabled = true;
        return false;
    }
    initialized = true;
    return true;
}

ATTRIBUTE_NO_SANITIZE_ALL
void Tracer::ClearMaps() {
    size_t map_sz = num_guards * sizeof(uint8_t);
    for (int i = 0; i < map_sz; ++i)
        feature_map[i] = 0;
}

ATTRIBUTE_NO_SANITIZE_ALL
inline void Tracer::InitializeGuards(uint32_t *start, uint32_t *stop) {
    if (start == stop || *start) return;  // Initialize only once.
    for (uint32_t *x = start; x < stop; x++)
        *x = ++num_guards;  // Guards should start from 1.
}

ATTRIBUTE_NO_SANITIZE_ALL
inline void Tracer::HandleTracePCGuard(uintptr_t pc, uint32_t* guard) {
    if (disabled)
        return;
    if (!initialized && !InitializeMaps())
        return;
    assert(*guard && "Null guard detected after initialization");

    uint32_t idx = *guard - 1;
    if (__builtin_add_overflow(feature_map[idx], 1, &feature_map[idx]))
        feature_map[idx] = UINT8_MAX;
}

} // namespace fuzzer


extern "C" {

ATTRIBUTE_NO_SANITIZE_ALL
void __sanitizer_cov_trace_pc_guard_init(uint32_t *start, uint32_t *stop) {
    fuzzer::CoverageTracer.InitializeGuards(start, stop);
}

ATTRIBUTE_NO_SANITIZE_ALL
void __sanitizer_cov_trace_pc_guard(uint32_t *guard) {
    uintptr_t PC = reinterpret_cast<uintptr_t>(GET_CALLER_PC());
    fuzzer::CoverageTracer.HandleTracePCGuard(PC, guard);
}

} // extern "C"
