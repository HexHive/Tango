// Compile binaries with -fsanitize-coverage={func, bb, edge},trace-pc-guard
// Minimum clang version: 13.0

#pragma once

#if !__has_feature(coverage_sanitizer)
#error Incompatible compiler! Please use Clang 13.0 or higher
#endif

#define GET_CALLER_PC() __builtin_return_address(0)