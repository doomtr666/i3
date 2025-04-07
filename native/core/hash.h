#pragma once

#include "bitset.h"

// MumurHash3 32-bit
static inline uint32_t i3_hash32(const void* key, uint32_t key_size, uint32_t seed)
{
    static const uint32_t c1 = 0xcc9e2d51;
    static const uint32_t c2 = 0x1b873593;
    static const uint32_t r1 = 15;
    static const uint32_t r2 = 13;
    static const uint32_t m = 5;
    static const uint32_t n = 0xe6546b64;

    uint32_t hash = seed;

    const int nblocks = key_size / sizeof(uint32_t);
    const uint32_t* blocks = (const uint32_t*)key;

    for (int i = 0; i < nblocks; i++)
    {
        uint32_t k = blocks[i];
        k *= c1;
        k = i3_bit_rol32(k, r1);
        k *= c2;

        hash ^= k;
        hash = i3_bit_rol32(hash, r2) * m + n;
    }

    const uint8_t* tail = (const uint8_t*)(blocks + nblocks);
    uint32_t k1 = 0;

    switch (key_size & 3)
    {
    case 3:
        k1 ^= tail[2] << 16;
    case 2:
        k1 ^= tail[1] << 8;
    case 1:
        k1 ^= tail[0];

        k1 *= c1;
        k1 = i3_bit_rol32(k1, r1);
        k1 *= c2;
        hash ^= k1;
    }

    hash ^= key_size;
    hash ^= (hash >> 16);
    hash *= 0x85ebca6b;
    hash ^= (hash >> 13);
    hash *= 0xc2b2ae35;
    hash ^= (hash >> 16);

    return hash;
}

