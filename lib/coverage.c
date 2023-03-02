#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

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

static const u8 count_hamming_bits[256] = {
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

static inline u32 hash32(const void *key, u32 len, u32 seed);

bool diff(u8 *set_arr, const u8 *coverage_map, u8 *set_map, \
    const size_t map_size, u32 *set_count, u32 *map_hash, const bool inplace)
{
  u8 kls;
  *set_count = 0;
  for (size_t i = 0; i < map_size; ++i) {
    kls = count_class_lookup8[coverage_map[i]];
    set_map[i] = (set_arr[i] | kls) ^ set_arr[i];
    *set_count += count_hamming_bits[set_map[i]];
    if (inplace)
      set_arr[i] |= kls;
  }
  *map_hash = hash32(set_map, map_size, HASH_CONST);
  return true;
}

bool apply(u8 *set_arr, const u8 *set_map, const size_t map_size)
{
  for (size_t i = 0; i < map_size; ++i)
    set_arr[i] ^= set_map[i];
  return true;
}

#define ROL64(_x, _r)  ((((u64)(_x)) << (_r)) | (((u64)(_x)) >> (64 - (_r))))

static inline u32 hash32(const void* key, u32 len, u32 seed)
{
  const u64* data = (u64*)key;
  u64 h1 = seed ^ len;

  rem = len & 0b111;
  len >>= 3;

  while (len--) {

    u64 k1 = *data++;
    key += 1<<3;

    k1 *= 0x87c37b91114253d5ULL;
    k1  = ROL64(k1, 31);
    k1 *= 0x4cf5ad432745937fULL;

    h1 ^= k1;
    h1  = ROL64(h1, 27);
    h1  = h1 * 5 + 0x52dce729;

  }

  u64 data_rem = 0;
  for (int i = 0; i < rem; ++i)
    data_rem += *((u8*)key++) << (8 * i);
  u64 k1 = data_rem;
  k1 *= 0x87c37b91114253d5ULL;
  k1  = ROL64(k1, 31);
  k1 *= 0x4cf5ad432745937fULL;
  h1 ^= k1;
  h1  = ROL64(h1, 27);
  h1  = h1 * 5 + 0x52dce729;

  h1 ^= h1 >> 33;
  h1 *= 0xff51afd7ed558ccdULL;
  h1 ^= h1 >> 33;
  h1 *= 0xc4ceb9fe1a85ec53ULL;
  h1 ^= h1 >> 33;

  return h1;
}