#pragma once

#include "common.h"

typedef uint64_t i3_bitset_t;
#define i3_bitset_bit_size 64
#define i3_bitset_shift 6

// declare bitset of size
#define i3_bitset(name, size) i3_bitset_t name[i3_bitset_array_size(size)]

// bitset size in bits to array of i3_bitset_t size
#define i3_bitset_array_size(size) (((size) + i3_bitset_bit_size - 1) / i3_bitset_bit_size)

// index in array for a position
#define i3_bitset_index(pos) ((pos) >> i3_bitset_shift)

// offset in one element of the bit
#define i3_bitset_offset(pos) ((pos) & ((i3_bitset_bit_size)-1))

// value zero for the bitset
#define i3_bitset_zero ((i3_bitset_t)0)

// value one for the bitset
#define i3_bitset_one ((i3_bitset_t)1)

// max value for the bitset (all one bits)
#define i3_bitset_max_val ((i3_bitset_t)-1)

// bit rotation
static inline uint32_t i3_bit_ror32(uint32_t x, uint32_t r)
{
#if I3_PLATFORM == I3_PLATFORM_WINDOWS
    return _rotr(x, r);
#else
    // naive implementation 
    return (x >> r) | (x << (32 - r));
#endif
}

static inline uint32_t i3_bit_rol32(uint32_t x, uint32_t r)
{
#if I3_PLATFORM == I3_PLATFORM_WINDOWS
    return _rotl(x, r);
#else
    // naive implementation 
    return (x << r) | (x >> (32 - r));
#endif
}

// count the number of bits set in a 64-bit integer
static inline uint32_t i3_bitcount64(uint64_t x)
{
#if I3_PLATFORM == I3_PLATFORM_WINDOWS
    return (uint32_t)__popcnt64(x);
#else
    // naive implementation
    uint32_t count;
    for (count = 0; x; count += x & 1, x >>= 1);
    return count;
#endif
}

static inline uint32_t i3_bit_scan_forward64(uint64_t x)
{
#if I3_PLATFORM == I3_PLATFORM_WINDOWS
    unsigned long index;
    if (_BitScanForward64(&index, x) > 0)
        return (uint32_t)index;
    return UINT32_MAX;
#else
    // naive implementation
    for (uint32_t i = 0; i < 64; ++i)
        if (x & (1ull << i))
            return i;
    return UINT32_MAX;
#endif
}

static inline uint32_t i3_bit_scan_reverse64(uint64_t x)
{
#if I3_PLATFORM == I3_PLATFORM_WINDOWS
    unsigned long index;
    if (_BitScanReverse64(&index, x) > 0)
        return (uint32_t)index;
    return UINT32_MAX;
#else
    // naive implementation
    for (int32_t i = sizeof(i3_bitset_t) - 1; i >= 0; --i)
        if (x & (1ull << i))
            return (uint32_t)i;
    return UINT32_MAX;
#endif
}

// init the bitset
static inline void i3_bitset_init(i3_bitset_t* b, uint32_t size, bool value)
{
    assert(b != NULL);

    i3_bitset_t v = value ? i3_bitset_max_val : 0;
    for (uint32_t i = 0; i < i3_bitset_array_size(size); i++)
        b[i] = v;
}

// clear the bitset
static inline void i3_bitset_clear(i3_bitset_t* b, uint32_t size)
{
    assert(b != NULL);

    i3_bitset_init(b, size, false);
}

// copy bitset
static inline void i3_bitset_copy(i3_bitset_t* d, const i3_bitset_t* s, uint32_t size)
{
    assert(d != NULL);
    assert(s != NULL);

    for (uint32_t i = 0; i < i3_bitset_array_size(size); i++)
        d[i] = s[i];
}

// test one bit of the bitset
static inline bool i3_bitset_test(const i3_bitset_t* b, uint32_t pos)
{
    assert(b != NULL);

    uint32_t index = i3_bitset_index(pos);
    uint32_t offset = i3_bitset_offset(pos);
    return (b[index] >> offset & i3_bitset_one) != 0;
}

// set one bit of the bitset
static inline void i3_bitset_set(i3_bitset_t* b, uint32_t pos)
{
    assert(b != NULL);

    uint32_t index = i3_bitset_index(pos);
    uint32_t offset = i3_bitset_offset(pos);
    b[index] |= i3_bitset_one << offset;
}

// reset one bit of the bitset
static inline void i3_bitset_reset(i3_bitset_t* b, uint32_t pos)
{
    assert(b != NULL);

    uint32_t index = i3_bitset_index(pos);
    uint32_t offset = i3_bitset_offset(pos);
    b[index] &= ~(i3_bitset_one << offset);
}

// equality
static inline bool i3_bitset_equals(const i3_bitset_t* b1, const i3_bitset_t* b2, uint32_t size)
{
    assert(b1 != NULL);
    assert(b2 != NULL);

    for (uint32_t i = 0; i < i3_bitset_array_size(size); i++)
        if (b1[i] != b2[i])
            return false;
    return true;
}

// count the number of bits set
static inline uint32_t i3_bitset_count(const i3_bitset_t* b, uint32_t size)
{
    assert(b != NULL);

    uint32_t count = 0;
    for (uint32_t i = 0; i < i3_bitset_array_size(size); i++)
        count += i3_bitcount64(b[i]);
    return count;
}

// find the first bit set
static inline uint32_t i3_bitset_find_first(const i3_bitset_t* b, uint32_t size)
{
    assert(b != NULL);

    for (uint32_t i = 0; i < i3_bitset_array_size(size); i++)
    {
        if (b[i] != 0)
            return i3_bit_scan_forward64(b[i]) + i * i3_bitset_bit_size;
    }
    return UINT32_MAX;
}

// find the last bit set
static inline uint32_t i3_bitset_find_last(const i3_bitset_t* b, uint32_t size)
{
    assert(b != NULL);

    for (int32_t i = i3_bitset_array_size(size); i >= 0; i--)
    {
        if (b[i - 1] != 0)
            return i3_bit_scan_reverse64(b[i - 1]) + (i - 1) * i3_bitset_bit_size;
    }
    return UINT32_MAX;
}

// bitwise operations
static inline void i3_bitset_not(i3_bitset_t* r, const i3_bitset_t* b, uint32_t size)
{
    assert(r != NULL);
    assert(b != NULL);

    for (uint32_t i = 0; i < i3_bitset_array_size(size); i++)
        r[i] = ~b[i];
}

static inline void i3_bitset_and(i3_bitset_t* r, const i3_bitset_t* b1, const i3_bitset_t* b2, uint32_t size)
{
    assert(r != NULL);
    assert(b1 != NULL);
    assert(b2 != NULL);

    for (uint32_t i = 0; i < i3_bitset_array_size(size); i++)
        r[i] = b1[i] & b2[i];
}

static inline void i3_bitset_or(i3_bitset_t* r, const i3_bitset_t* b1, const i3_bitset_t* b2, uint32_t size)
{
    assert(r != NULL);
    assert(b1 != NULL);
    assert(b2 != NULL);

    for (uint32_t i = 0; i < i3_bitset_array_size(size); i++)
        r[i] = b1[i] | b2[i];
}

static inline void i3_bitset_xor(i3_bitset_t* r, const i3_bitset_t* b1, const i3_bitset_t* b2, uint32_t size)
{
    assert(r != NULL);
    assert(b1 != NULL);
    assert(b2 != NULL);

    for (uint32_t i = 0; i < i3_bitset_array_size(size); i++)
        r[i] = b1[i] ^ b2[i];
}
