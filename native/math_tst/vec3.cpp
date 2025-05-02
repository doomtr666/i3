#include <gtest/gtest.h>

extern "C"
{
#include "native/math/vec.h"
}

TEST(vec3, abs)
{
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_abs({-1, 2, -3}), {1, 2, 3}, 1e-6f));
}

TEST(vec3, neg)
{
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_neg({1, -2, 3}), {-1, 2, -3}, 1e-6f));
}

TEST(vec3, add)
{
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_add({1, 2, 3}, {4, 5, 6}), {5, 7, 9}, 1e-6f));
}

TEST(vec3, sub)
{
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_sub({1, 2, 3}, {4, 5, 6}), {-3, -3, -3}, 1e-6f));
}

TEST(vec3, mul)
{
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_mul({1, 2, 3}, {4, 5, 6}), {4, 10, 18}, 1e-6f));
}

TEST(vec3, div)
{
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_div({1, 2, 3}, {4, 5, 6}), {0.25f, 0.4f, 0.5f}, 1e-6f));
}

TEST(vec3, scale)
{
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_scale({1, 2, 3}, 4), {4, 8, 12}, 1e-6f));
}

TEST(vec3, dot)
{
    EXPECT_FLOAT_EQ(i3_vec3_dot({1, 2, 3}, {4, 5, 6}), 32.0f);
}

TEST(vec3, len2)
{
    EXPECT_FLOAT_EQ(i3_vec3_len2({3, 4, 5}), 50.0f);
}

TEST(vec3, len)
{
    EXPECT_FLOAT_EQ(i3_vec3_len({3, 4, 5}), 7.071068f);
}

TEST(vec3, normalize)
{
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_normalize({3, 4, 5}), {0.424264f, 0.565685f, 0.707107f}, 1e-6f));
}

TEST(vec3, min)
{
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_min({1, 2, 3}, {4, 5, 6}), {1, 2, 3}, 1e-6f));
}

TEST(vec3, max)
{
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_max({1, 2, 3}, {4, 5, 6}), {4, 5, 6}, 1e-6f));
}

TEST(vec3, clamp)
{
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_clamp({1, 2, 3}, 0, 2), {1, 2, 2}, 1e-6f));
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_clamp({-1, -2, -3}, 0, 2), {0, 0, 0}, 1e-6f));
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_clamp({3, 4, 5}, 0, 2), {2, 2, 2}, 1e-6f));
}

TEST(vec3, saturate)
{
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_saturate({-1, -2, -3}), {0, 0, 0}, 1e-6f));
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_saturate({0.5f, 0.5f, 0.5f}), {0.5f, 0.5f, 0.5f}, 1e-6f));
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_saturate({1, 2, 3}), {1, 1, 1}, 1e-6f));
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_saturate({4, 5, 6}), {1, 1, 1}, 1e-6f));
}
