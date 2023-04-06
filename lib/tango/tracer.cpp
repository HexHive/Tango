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

template <class T>
struct SharedMemoryObject {
    T *pObj;

    ATTRIBUTE_NO_SANITIZE_ALL
    SharedMemoryObject(const char *name, size_t size,
            int oflags = O_RDWR | O_CREAT | O_TRUNC,
            mode_t omode = S_IRUSR | S_IWUSR,
            int mprot = PROT_READ | PROT_WRITE)
            : uuid(getppid())
    {
        char path[PATH_MAX];
        snprintf(path, PATH_MAX, "/tango_%s_%lu", name, uuid);

        int fd = shm_open(path, oflags, omode);
        if (fd == -1)
            throw std::runtime_error("Failed to open shm region");
        if (ftruncate(fd, size) == -1)
            throw std::runtime_error("Failed to truncate shm region");
        pObj = (T*)mmap(NULL, size, mprot, MAP_SHARED, fd, 0);
        close(fd);
        if (!pObj)
            throw std::runtime_error("Failed to mmap shm_region");
    }

private:
    uint64_t uuid;
};

// function definitions

ATTRIBUTE_NO_SANITIZE_ALL
Tracer::Tracer() {
    initialized = false;
    disabled = false;
    num_guards = 0;
}

ATTRIBUTE_NO_SANITIZE_ALL
bool Tracer::InitializeMaps() {
    try {
        size_t feature_size = num_guards * sizeof(uint8_t);
        feature_map = SharedMemoryObject<uint8_t>(
            "cov", feature_size).pObj;

        uint32_t *map_size = SharedMemoryObject<uint32_t>(
            "size", sizeof(uint32_t)).pObj;
        *map_size = feature_size;
        munmap(map_size, sizeof(uint32_t));
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

template <class T>
ATTRIBUTE_NO_SANITIZE_ALL
inline void Tracer::HandleCmp(uintptr_t PC, T Arg1, T Arg2) {
    uint64_t ArgXor = Arg1 ^ Arg2;
    switch (sizeof(T)) {
    case 2:
        TORC2.Insert(Arg1, Arg2);
    case 4:
        TORC4.Insert(Arg1, Arg2);
    case 8:
        TORC8.Insert(Arg1, Arg2);
    default:
        return;
    }
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

#ifdef USE_CMPLOG
ATTRIBUTE_NO_SANITIZE_ALL
void __sanitizer_cov_trace_cmp8(uint64_t Arg1, uint64_t Arg2) {
  uintptr_t PC = reinterpret_cast<uintptr_t>(GET_CALLER_PC());
  fuzzer::CoverageTracer.HandleCmp(PC, Arg1, Arg2);
}

ATTRIBUTE_NO_SANITIZE_ALL
void __sanitizer_cov_trace_const_cmp8(uint64_t Arg1, uint64_t Arg2) {
  uintptr_t PC = reinterpret_cast<uintptr_t>(GET_CALLER_PC());
  fuzzer::CoverageTracer.HandleCmp(PC, Arg1, Arg2);
}

ATTRIBUTE_NO_SANITIZE_ALL
void __sanitizer_cov_trace_cmp4(uint32_t Arg1, uint32_t Arg2) {
  uintptr_t PC = reinterpret_cast<uintptr_t>(GET_CALLER_PC());
  fuzzer::CoverageTracer.HandleCmp(PC, Arg1, Arg2);
}

ATTRIBUTE_NO_SANITIZE_ALL
void __sanitizer_cov_trace_const_cmp4(uint32_t Arg1, uint32_t Arg2) {
  uintptr_t PC = reinterpret_cast<uintptr_t>(GET_CALLER_PC());
  fuzzer::CoverageTracer.HandleCmp(PC, Arg1, Arg2);
}

ATTRIBUTE_NO_SANITIZE_ALL
void __sanitizer_cov_trace_cmp2(uint16_t Arg1, uint16_t Arg2) {
  uintptr_t PC = reinterpret_cast<uintptr_t>(GET_CALLER_PC());
  fuzzer::CoverageTracer.HandleCmp(PC, Arg1, Arg2);
}

ATTRIBUTE_NO_SANITIZE_ALL
void __sanitizer_cov_trace_const_cmp2(uint16_t Arg1, uint16_t Arg2) {
  uintptr_t PC = reinterpret_cast<uintptr_t>(GET_CALLER_PC());
  fuzzer::CoverageTracer.HandleCmp(PC, Arg1, Arg2);
}

ATTRIBUTE_NO_SANITIZE_ALL
void __sanitizer_cov_trace_cmp1(uint8_t Arg1, uint8_t Arg2) {
  uintptr_t PC = reinterpret_cast<uintptr_t>(GET_CALLER_PC());
  fuzzer::CoverageTracer.HandleCmp(PC, Arg1, Arg2);
}

ATTRIBUTE_NO_SANITIZE_ALL
void __sanitizer_cov_trace_const_cmp1(uint8_t Arg1, uint8_t Arg2) {
  uintptr_t PC = reinterpret_cast<uintptr_t>(GET_CALLER_PC());
  fuzzer::CoverageTracer.HandleCmp(PC, Arg1, Arg2);
}

ATTRIBUTE_NO_SANITIZE_ALL
void __sanitizer_cov_trace_switch(uint64_t Val, uint64_t *Cases) {
  uint64_t N = Cases[0];
  uint64_t ValSizeInBits = Cases[1];
  uint64_t *Vals = Cases + 2;
  // Skip the most common and the most boring case: all switch values are small.
  // We may want to skip this at compile-time, but it will make the
  // instrumentation less general.
  if (Vals[N - 1]  < 256)
    return;
  // Also skip small inputs values, they won't give good signal.
  if (Val < 256)
    return;
  uintptr_t PC = reinterpret_cast<uintptr_t>(GET_CALLER_PC());
  size_t i;
  uint64_t Smaller = 0;
  uint64_t Larger = ~(uint64_t)0;
  // Find two switch values such that Smaller < Val < Larger.
  // Use 0 and 0xfff..f as the defaults.
  for (i = 0; i < N; i++) {
    if (Val < Vals[i]) {
      Larger = Vals[i];
      break;
    }
    if (Val > Vals[i]) Smaller = Vals[i];
  }

  // Apply HandleCmp to {Val,Smaller} and {Val, Larger},
  // use i as the PC modifier for HandleCmp.
  if (ValSizeInBits == 16) {
    fuzzer::CoverageTracer.HandleCmp(PC + 2 * i, static_cast<uint16_t>(Val),
                          (uint16_t)(Smaller));
    fuzzer::CoverageTracer.HandleCmp(PC + 2 * i + 1, static_cast<uint16_t>(Val),
                          (uint16_t)(Larger));
  } else if (ValSizeInBits == 32) {
    fuzzer::CoverageTracer.HandleCmp(PC + 2 * i, static_cast<uint32_t>(Val),
                          (uint32_t)(Smaller));
    fuzzer::CoverageTracer.HandleCmp(PC + 2 * i + 1, static_cast<uint32_t>(Val),
                          (uint32_t)(Larger));
  } else {
    fuzzer::CoverageTracer.HandleCmp(PC + 2*i, Val, Smaller);
    fuzzer::CoverageTracer.HandleCmp(PC + 2*i + 1, Val, Larger);
  }
}
#endif

} // extern "C"
