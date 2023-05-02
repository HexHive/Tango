#pragma once

#define GET_CALLER_PC() __builtin_return_address(0)

#define ATTRIBUTE_NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))
#define ATTRIBUTE_NO_SANITIZE_COVERAGE __attribute__((no_sanitize("coverage")))

#if __has_feature(address_sanitizer) && __has_feature(coverage_sanitizer)
#define ATTRIBUTE_NO_SANITIZE_ALL \
    ATTRIBUTE_NO_SANITIZE_ADDRESS ATTRIBUTE_NO_SANITIZE_COVERAGE
#elif __has_feature(address_sanitizer)
#define ATTRIBUTE_NO_SANITIZE_ALL ATTRIBUTE_NO_SANITIZE_ADDRESS
#elif __has_feature(coverage_sanitizer)
#define ATTRIBUTE_NO_SANITIZE_ALL ATTRIBUTE_NO_SANITIZE_COVERAGE
#else
#define ATTRIBUTE_NO_SANITIZE_ALL
#endif

#ifdef __x86_64
#if __has_attribute(target)
#define ATTRIBUTE_TARGET_POPCNT __attribute__((target("popcnt")))
#define HAS_POPCNT
#else
#define ATTRIBUTE_TARGET_POPCNT
#endif
#else
#define ATTRIBUTE_TARGET_POPCNT
#endif

#ifndef HAS_POPCNT
static const unsigned char popcount_hamming_8[256] = {
  0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3,
  3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
  3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2,
  2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5,
  3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
  5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3,
  2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4,
  4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4,
  4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6,
  5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5,
  5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};
ATTRIBUTE_NO_SANITIZE_ALL
static inline int Popcount1(unsigned char X) { return popcount_hamming_8[X]; }
ATTRIBUTE_NO_SANITIZE_ALL
static inline int Popcount2(unsigned short X) {
    return \
        popcount_hamming_8[((unsigned char *)&X)[0]] + \
        popcount_hamming_8[((unsigned char *)&X)[1]];
}
ATTRIBUTE_NO_SANITIZE_ALL
static inline int Popcount4(unsigned int X) {
    return \
        popcount_hamming_8[((unsigned char *)&X)[0]] + \
        popcount_hamming_8[((unsigned char *)&X)[1]] + \
        popcount_hamming_8[((unsigned char *)&X)[2]] + \
        popcount_hamming_8[((unsigned char *)&X)[3]];
}
ATTRIBUTE_NO_SANITIZE_ALL
static inline int Popcount8(unsigned long long X) {
    return \
        popcount_hamming_8[((unsigned char *)&X)[0]] + \
        popcount_hamming_8[((unsigned char *)&X)[1]] + \
        popcount_hamming_8[((unsigned char *)&X)[2]] + \
        popcount_hamming_8[((unsigned char *)&X)[3]] + \
        popcount_hamming_8[((unsigned char *)&X)[4]] + \
        popcount_hamming_8[((unsigned char *)&X)[5]] + \
        popcount_hamming_8[((unsigned char *)&X)[6]] + \
        popcount_hamming_8[((unsigned char *)&X)[7]];
}
#else
ATTRIBUTE_TARGET_POPCNT ATTRIBUTE_NO_SANITIZE_ALL
static inline int Popcount1(unsigned char X) { return __builtin_popcountll(X); }
ATTRIBUTE_TARGET_POPCNT ATTRIBUTE_NO_SANITIZE_ALL
static inline int Popcount2(unsigned short X) { return __builtin_popcountll(X); }
ATTRIBUTE_TARGET_POPCNT ATTRIBUTE_NO_SANITIZE_ALL
static inline int Popcount4(unsigned int X) { return __builtin_popcountll(X); }
ATTRIBUTE_TARGET_POPCNT ATTRIBUTE_NO_SANITIZE_ALL
static inline int Popcount8(unsigned long long X) { return __builtin_popcountll(X); }
#endif