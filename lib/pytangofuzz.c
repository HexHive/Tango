#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

#define HASH_CONST          0xa5b35705

#ifndef SKIP_COUNTS
static const u8 count_class_lookup8[256] = {
  [0]           = 0,
  [1]           = 1,
  [2]           = 2,
  [3]           = 4,
  [4 ... 7]     = 8,
  [8 ... 15]    = 16,
  [16 ... 31]   = 32,
  [32 ... 127]  = 64,
  [128 ... 255] = 128
};
#else
static const u8 count_class_lookup8[256] = {
  [0]           = 0,
  [1 ... 255]   = 1
};
#endif

static inline u32 hash32(const void* key, u32 len, u32 seed);
static inline void classify_counts(u8 *binned, u8 *map, size_t length);

u32 hash_cov(u8 *address, size_t length)
{
    u8 *binned = calloc(length, sizeof(u8));
    classify_counts(binned, address, length);
    u32 hash = hash32(binned, length, HASH_CONST);
    free(binned);
    return hash;
}

static inline void classify_counts(u8 *binned, u8 *map, size_t length) {
    while (length--) {
        *binned = count_class_lookup8[*map];
        map++; binned++;
    }
}

#define ROL64(_x, _r)  ((((u64)(_x)) << (_r)) | (((u64)(_x)) >> (64 - (_r))))

static inline u32 hash32(const void* key, u32 len, u32 seed)
{
  const u64* data = (u64*)key;
  u64 h1 = seed ^ len;

  len >>= 3;

  while (len--) {

    u64 k1 = *data++;

    k1 *= 0x87c37b91114253d5ULL;
    k1  = ROL64(k1, 31);
    k1 *= 0x4cf5ad432745937fULL;

    h1 ^= k1;
    h1  = ROL64(h1, 27);
    h1  = h1 * 5 + 0x52dce729;

  }

  h1 ^= h1 >> 33;
  h1 *= 0xff51afd7ed558ccdULL;
  h1 ^= h1 >> 33;
  h1 *= 0xc4ceb9fe1a85ec53ULL;
  h1 ^= h1 >> 33;

  return h1;
}