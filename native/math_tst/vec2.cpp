#include <gtest/gtest.h>

extern "C"
{
#include "native/math/vec.h"
}

TEST(vec2, abs)
{
    EXPECT_TRUE(i3_vec2_eq(i3_vec2_abs({-1, 2}), {1, 2}, 1e-6f));
}

TEST(vec2, neg)
{
    EXPECT_TRUE(i3_vec2_eq(i3_vec2_neg({1, -2}), {-1, 2}, 1e-6f));
}

TEST(vec2, add)
{
    EXPECT_TRUE(i3_vec2_eq(i3_vec2_add({1, 2}, {3, 4}), {4, 6}, 1e-6f));
}

TEST(vec2, sub)
{
    EXPECT_TRUE(i3_vec2_eq(i3_vec2_sub({1, 2}, {3, 4}), {-2, -2}, 1e-6f));
}

TEST(vec2, mul)
{
    EXPECT_TRUE(i3_vec2_eq(i3_vec2_mul({1, 2}, {3, 4}), {3, 8}, 1e-6f));
}

TEST(vec2, div)
{
    EXPECT_TRUE(i3_vec2_eq(i3_vec2_div({1, 2}, {3, 4}), {1.0f / 3.0f, 0.5f}, 1e-6f));
}

TEST(vec2, scale)
{
    EXPECT_TRUE(i3_vec2_eq(i3_vec2_scale({1, 2}, 3), {3, 6}, 1e-6f));
}

TEST(vec2, dot)
{
    EXPECT_FLOAT_EQ(i3_vec2_dot({1, 2}, {3, 4}), 11.0f);
}

TEST(vec2, len2)
{
    EXPECT_FLOAT_EQ(i3_vec2_len2({3, 4}), 25.0f);
}

TEST(vec2, len)
{
    EXPECT_FLOAT_EQ(i3_vec2_len({3, 4}), 5.0f);
}

TEST(vec2, normalize)
{
    EXPECT_TRUE(i3_vec2_eq(i3_vec2_normalize({3, 4}), {0.6f, 0.8f}, 1e-6f));
}

TEST(vec2, min)
{
    EXPECT_TRUE(i3_vec2_eq(i3_vec2_min({1, 2}, {3, 4}), {1, 2}, 1e-6f));
}

TEST(vec2, max)
{
    EXPECT_TRUE(i3_vec2_eq(i3_vec2_max({1, 2}, {3, 4}), {3, 4}, 1e-6f));
}

TEST(vec2, clamp)
{
    EXPECT_TRUE(i3_vec2_eq(i3_vec2_clamp({1, 2}, {0, 0}, {3, 3}), {1, 2}, 1e-6f));
    EXPECT_TRUE(i3_vec2_eq(i3_vec2_clamp({-1, 4}, {0, 0}, {3, 3}), {0, 3}, 1e-6f));
    EXPECT_TRUE(i3_vec2_eq(i3_vec2_clamp({4, -1}, {0, 0}, {3, 3}), {3, 0}, 1e-6f));
}

TEST(vec2, saturate)
{
    EXPECT_TRUE(i3_vec2_eq(i3_vec2_saturate({1, 2}), {1, 1}, 1e-6f));
    EXPECT_TRUE(i3_vec2_eq(i3_vec2_saturate({-1, 4}), {0, 1}, 1e-6f));
    EXPECT_TRUE(i3_vec2_eq(i3_vec2_saturate({4, -1}), {1, 0}, 1e-6f));
}