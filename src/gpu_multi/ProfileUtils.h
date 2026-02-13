#pragma once
#include <nvtx3/nvToolsExt.h>

// Simple RAII wrapper for NVTX ranges
class TraceScope {
public:
    TraceScope(const char* name) { nvtxRangePush(name); }
    ~TraceScope() { nvtxRangePop(); }
};

// Macro to make usage cleaner:
// Usage: PROFILE_SCOPE("My Function Name");
#define PROFILE_SCOPE(name) TraceScope _trace_scope_instance(name)