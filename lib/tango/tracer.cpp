#if !__has_feature(coverage_sanitizer)
#error Incompatible compiler! Please use Clang 13.0 or higher
#endif

#include "common.h"
#include "tracer.h"
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <assert.h>
#include <stdexcept>
#include <type_traits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <linux/limits.h>

namespace fuzzer {

Tracer CoverageTracer;

template <class T>
struct SharedMemoryObject {
    T *pMap;

    ATTRIBUTE_NO_SANITIZE_ALL
    SharedMemoryObject(const char *name, size_t size,
            int oflags = O_RDWR | O_CREAT | O_TRUNC,
            mode_t omode = S_IRUSR | S_IWUSR,
            int mprot = PROT_READ | PROT_WRITE)
            : uuid(getppid())
    {
        char path[PATH_MAX];
        snprintf(path, PATH_MAX, "/tango_%s_%lu", name, uuid);

        size_t obj_size = sizeof(size) + size;
        int fd = shm_open(path, oflags, omode);
        if (fd == -1)
            throw std::runtime_error("Failed to open shm region");
        if (ftruncate(fd, obj_size) == -1)
            throw std::runtime_error("Failed to truncate shm region");
        pObj = mmap(NULL, obj_size, mprot, MAP_SHARED, fd, 0);
        close(fd);
        if (!pObj)
            throw std::runtime_error("Failed to mmap shm_region");
        pMap = (T*)((uintptr_t)pObj + sizeof(size));
        // set the size field to the requested size
        *(size_t *)pObj = size;
    }

private:
    uint64_t uuid;
    void *pObj;
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
        feature_map = SharedMemoryObject<uint8_t>(
            "cov", num_guards * sizeof(uint8_t)).pMap;
        pc_map = SharedMemoryObject<uintptr_t>(
            "pc", num_guards * sizeof(uintptr_t)).pMap;

#ifdef USE_CMPLOG
        TORC1 = SharedMemoryObject<std::remove_pointer<decltype(TORC1)>::type>(
            "torc1", sizeof(*TORC1)).pMap;
        TORC2 = SharedMemoryObject<std::remove_pointer<decltype(TORC2)>::type>(
            "torc2", sizeof(*TORC2)).pMap;
        TORC4 = SharedMemoryObject<std::remove_pointer<decltype(TORC4)>::type>(
            "torc4", sizeof(*TORC4)).pMap;
        TORC8 = SharedMemoryObject<std::remove_pointer<decltype(TORC8)>::type>(
            "torc8", sizeof(*TORC8)).pMap;

        memset(TORC1, 0, sizeof(*TORC1));
        memset(TORC2, 0, sizeof(*TORC2));
        memset(TORC4, 0, sizeof(*TORC4));
        memset(TORC8, 0, sizeof(*TORC8));
#endif
    } catch (...) {
        disabled = true;
        return false;
    }
    initialized = true;
    return true;
}

ATTRIBUTE_NO_SANITIZE_ALL
void Tracer::ClearMaps() {
    memset(feature_map, 0, num_guards);

#ifdef USE_CMPLOG
    TORC1->Clear();
    TORC2->Clear();
    TORC4->Clear();
    TORC8->Clear();
#endif
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
    pc_map[idx] = pc - 1;
}

#ifdef USE_CMPLOG
template <class T>
ATTRIBUTE_NO_SANITIZE_ALL
inline void Tracer::HandleCmp(uintptr_t PC, T Arg1, T Arg2) {
    if (Arg1 == Arg2 || (Arg1 <= 0xff && Arg2 <= 0xff))
        return;
    // this is optimized out by the compiler
    switch (sizeof(T)) {
    case 1:
        // for now, this is unreachable due to the 0xff check above
        TORC1->Insert(Arg1, Arg2);
    case 2:
        TORC2->Insert(Arg1, Arg2);
    case 4:
        TORC4->Insert(Arg1, Arg2);
    case 8:
        TORC8->Insert(Arg1, Arg2);
    default:
        return;
    }
}
#endif

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
