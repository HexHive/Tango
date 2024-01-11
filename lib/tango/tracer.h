#pragma once

#include <stdint.h>
#include <stdlib.h>

namespace fuzzer {

template<class T, size_t kCapacityT>
struct TableOfRecentCompares {
    struct Pair {
        T A, B;
    };

    static const size_t kCapacity = kCapacityT;
    size_t Length = 0;
    size_t LastIdx = 0;
    Pair Table[kCapacity];

    void Insert(const T &Arg1, const T &Arg2) {
        Table[LastIdx].A = Arg1;
        Table[LastIdx].B = Arg2;
        LastIdx = (LastIdx + 1) % kCapacity;
        if (Length < kCapacity)
            ++Length;
    }

    void Clear() {
        Length = LastIdx = 0;
    }
};

class Tracer {
public:
    Tracer();

    bool InitializeMaps();
    void ClearMaps();
    void InitializeGuards(uint32_t *start, uint32_t *stop);
    void HandleTracePCGuard(uintptr_t pc, uint32_t* guard);
#ifdef USE_CMPLOG
    template <class T> void HandleCmp(uintptr_t PC, T Arg1, T Arg2);
#endif

private:
    bool initialized;
    bool disabled;
    size_t num_guards;
    uint8_t *feature_map;
    uintptr_t *pc_map;
    bool ReshapeMaps();
    void ReshapMapsUnmap(void *old_map, void *new_map);

#ifdef USE_CMPLOG
    TableOfRecentCompares<uint8_t, 1024> *TORC1;
    TableOfRecentCompares<uint16_t, 1024> *TORC2;
    TableOfRecentCompares<uint32_t, 1024> *TORC4;
    TableOfRecentCompares<uint64_t, 1024> *TORC8;
#endif
};


} // namespace fuzzer

extern "C" {
    extern fuzzer::Tracer CoverageTracer;
}