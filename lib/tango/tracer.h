#pragma once

#include <stdint.h>
#include <stdlib.h>

namespace fuzzer {

class Tracer {
public:
    Tracer();

    void InitializeMaps();
    void ClearMaps();
    void InitializeGuards(uint32_t *start, uint32_t *stop);
    void HandleTracePCGuard(uintptr_t pc, uint32_t* guard);

private:
    bool initialized;
    bool disabled;
    size_t num_guards;
    uint8_t *feature_map;
};

extern Tracer CoverageTracer;

} // namespace fuzzer
