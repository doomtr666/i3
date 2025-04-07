#include "gtest/gtest.h"

extern "C"
{
#include "native/core/bitset.h"
}

TEST(bitset, bitcount64)
{
    EXPECT_EQ(i3_bitcount64(0), 0);
    EXPECT_EQ(i3_bitcount64(UINT64_MAX), 64);

    // check all bits
    for (uint64_t i = 0; i < 64; ++i)
    {
        uint64_t value = 1ull << i;
        EXPECT_EQ(i3_bitcount64(value), 1);
    }
}

TEST(bitset, bit_scan_forward64)
{
    EXPECT_EQ(i3_bit_scan_forward64(0), UINT32_MAX);

    for (uint64_t i = 0; i < 64; ++i)
    {
        uint64_t value = 1ull << i;
        EXPECT_EQ(i3_bit_scan_forward64(value), i);
    }
}

TEST(bitset, bit_scan_reverse64)
{
    EXPECT_EQ(i3_bit_scan_reverse64(0), UINT32_MAX);

    for (uint64_t i = 0; i < 64; ++i)
    {
        uint64_t value = 1ull << i;
        EXPECT_EQ(i3_bit_scan_reverse64(value), i);
    }
}

TEST(bitset, init)
{
    i3_bitset(bs, 256);

    i3_bitset_init(bs, 256, true);

    // check all bits
    for (uint32_t i = 0; i < 256; ++i)
        EXPECT_EQ(i3_bitset_test(bs, i), true);

    i3_bitset_init(bs, 256, false);

    // check all bits
    for (uint32_t i = 0; i < 256; ++i)
        EXPECT_EQ(i3_bitset_test(bs, i), false);
}

TEST(bitset, clear)
{
    i3_bitset(bs, 256);

    i3_bitset_init(bs, 256, true);

    // check all bits
    for (uint32_t i = 0; i < 256; ++i)
        EXPECT_EQ(i3_bitset_test(bs, i), true);

    i3_bitset_clear(bs, 256);

    // check all bits
    for (uint32_t i = 0; i < 256; ++i)
        EXPECT_EQ(i3_bitset_test(bs, i), false);
}

TEST(bitset, copy)
{
    i3_bitset(bs1, 256);
    i3_bitset(bs2, 256);

    i3_bitset_init(bs1, 256, true);
    i3_bitset_copy(bs2, bs1, 256);

    // check all bits
    for (uint32_t i = 0; i < 256; ++i)
        EXPECT_EQ(i3_bitset_test(bs2, i), true);

    i3_bitset_clear(bs1, 256);
    i3_bitset_copy(bs2, bs1, 256);

    // check all bits
    for (uint32_t i = 0; i < 256; ++i)
        EXPECT_EQ(i3_bitset_test(bs2, i), false);
}

TEST(bitset, set)
{
    i3_bitset(bs, 256);

    i3_bitset_init(bs, 256, false);

    i3_bitset_set(bs, 0);
    EXPECT_EQ(i3_bitset_test(bs, 0), true);

    i3_bitset_set(bs, 255);
    EXPECT_EQ(i3_bitset_test(bs, 255), true);

    i3_bitset_set(bs, 128);
    EXPECT_EQ(i3_bitset_test(bs, 128), true);
}

TEST(bitset, reset)
{
    i3_bitset(bs, 256);

    i3_bitset_init(bs, 256, true);

    i3_bitset_reset(bs, 0);
    EXPECT_EQ(i3_bitset_test(bs, 0), false);

    i3_bitset_reset(bs, 255);
    EXPECT_EQ(i3_bitset_test(bs, 255), false);

    i3_bitset_reset(bs, 128);
    EXPECT_EQ(i3_bitset_test(bs, 128), false);
}

TEST(bitset, equals)
{
    i3_bitset(bs1, 256);
    i3_bitset(bs2, 256);

    i3_bitset_init(bs1, 256, true);
    i3_bitset_init(bs2, 256, true);

    EXPECT_EQ(i3_bitset_equals(bs1, bs2, 256), true);

    i3_bitset_reset(bs1, 0);
    EXPECT_EQ(i3_bitset_equals(bs1, bs2, 256), false);

    i3_bitset_reset(bs2, 0);
    EXPECT_EQ(i3_bitset_equals(bs1, bs2, 256), true);
}

TEST(bitset, bitcount)
{
    i3_bitset(bs, 256);

    i3_bitset_init(bs, 256, true);

    EXPECT_EQ(i3_bitset_count(bs, 256), 256);

    i3_bitset_reset(bs, 0);
    EXPECT_EQ(i3_bitset_count(bs, 256), 255);

    i3_bitset_reset(bs, 255);
    EXPECT_EQ(i3_bitset_count(bs, 256), 254);

    i3_bitset_reset(bs, 128);
    EXPECT_EQ(i3_bitset_count(bs, 256), 253);
}

TEST(bitset, find_first)
{
    i3_bitset(bs, 256);

    i3_bitset_init(bs, 256, true);

    EXPECT_EQ(i3_bitset_find_first(bs, 256), 0);

    i3_bitset_reset(bs, 0);
    EXPECT_EQ(i3_bitset_find_first(bs, 256), 1);

    i3_bitset_reset(bs, 255);
    EXPECT_EQ(i3_bitset_find_first(bs, 256), 1);

    i3_bitset_reset(bs, 1);
    EXPECT_EQ(i3_bitset_find_first(bs, 256), 2);
}

TEST(bitset, find_last)
{
    i3_bitset(bs, 256);

    i3_bitset_init(bs, 256, true);

    EXPECT_EQ(i3_bitset_find_last(bs, 256), 255);

    i3_bitset_reset(bs, 255);
    EXPECT_EQ(i3_bitset_find_last(bs, 256), 254);

    i3_bitset_reset(bs, 0);
    EXPECT_EQ(i3_bitset_find_last(bs, 256), 254);

    i3_bitset_reset(bs, 254);
    EXPECT_EQ(i3_bitset_find_last(bs, 256), 253);
}

TEST(bitset, op_not)
{
    i3_bitset(bs1, 256);
    i3_bitset(bs2, 256);

    i3_bitset_init(bs1, 256, true);

    i3_bitset_not(bs2, bs1, 256);

    // check all bits
    for (uint32_t i = 0; i < 256; ++i)
        EXPECT_EQ(i3_bitset_test(bs2, i), false);

    i3_bitset_not(bs1, bs2, 256);

    // check all bits
    for (uint32_t i = 0; i < 256; ++i)
        EXPECT_EQ(i3_bitset_test(bs1, i), true);
}

TEST(bitset, op_and)
{
    i3_bitset(bs1, 256);
    i3_bitset(bs2, 256);
    i3_bitset(bs3, 256);

    i3_bitset_init(bs1, 256, true);
    i3_bitset_init(bs2, 256, true);

    i3_bitset_and(bs3, bs1, bs2, 256);

    // check all bits
    for (uint32_t i = 0; i < 256; ++i)
        EXPECT_EQ(i3_bitset_test(bs3, i), true);

    i3_bitset_clear(bs1, 256);
    i3_bitset_and(bs3, bs1, bs2, 256);

    // check all bits
    for (uint32_t i = 0; i < 256; ++i)
        EXPECT_EQ(i3_bitset_test(bs3, i), false);
}

TEST(bitset, op_or )
{
    i3_bitset(bs1, 256);
    i3_bitset(bs2, 256);
    i3_bitset(bs3, 256);

    i3_bitset_init(bs1, 256, true);
    i3_bitset_init(bs2, 256, true);

    i3_bitset_or(bs3, bs1, bs2, 256);

    // check all bits
    for (uint32_t i = 0; i < 256; ++i)
        EXPECT_EQ(i3_bitset_test(bs3, i), true);

    i3_bitset_clear(bs1, 256);
    i3_bitset_or(bs3, bs1, bs2, 256);

    // check all bits
    for (uint32_t i = 0; i < 256; ++i)
        EXPECT_EQ(i3_bitset_test(bs3, i), true);

    i3_bitset_clear(bs2, 256);
    i3_bitset_or(bs3, bs1, bs2, 256);

    // check all bits
    for (uint32_t i = 0; i < 256; ++i)
        EXPECT_EQ(i3_bitset_test(bs3, i), false);
}

TEST(bitset, op_xor)
{
    i3_bitset(bs1, 256);
    i3_bitset(bs2, 256);
    i3_bitset(bs3, 256);

    i3_bitset_init(bs1, 256, true);
    i3_bitset_init(bs2, 256, true);

    i3_bitset_xor(bs3, bs1, bs2, 256);

    // check all bits
    for (uint32_t i = 0; i < 256; ++i)
        EXPECT_EQ(i3_bitset_test(bs3, i), false);

    i3_bitset_clear(bs1, 256);
    i3_bitset_xor(bs3, bs1, bs2, 256);

    // check all bits
    for (uint32_t i = 0; i < 256; ++i)
        EXPECT_EQ(i3_bitset_test(bs3, i), true);

    i3_bitset_clear(bs2, 256);
    i3_bitset_xor(bs3, bs1, bs2, 256);

    // check all bits
    for (uint32_t i = 0; i < 256; ++i)
        EXPECT_EQ(i3_bitset_test(bs3, i), false);
}