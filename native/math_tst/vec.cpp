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

// vec3

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
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_clamp({1, 2, 3}, {0, 0, 0}, {2, 2, 2}), {1, 2, 2}, 1e-6f));
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_clamp({-1, -2, -3}, {0, 0, 0}, {2, 2, 2}), {0, 0, 0}, 1e-6f));
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_clamp({3, 4, 5}, {0, 0, 0}, {2, 2, 2}), {2, 2, 2}, 1e-6f));
}

TEST(vec3, saturate)
{
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_saturate({-1, -2, -3}), {0, 0, 0}, 1e-6f));
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_saturate({0.5f, 0.5f, 0.5f}), {0.5f, 0.5f, 0.5f}, 1e-6f));
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_saturate({1, 2, 3}), {1, 1, 1}, 1e-6f));
    EXPECT_TRUE(i3_vec3_eq(i3_vec3_saturate({4, 5, 6}), {1, 1, 1}, 1e-6f));
}

// vec4

TEST(vec4, abs)
{
    EXPECT_TRUE(i3_vec4_eq(i3_vec4_abs({-1, 2, -3, 4}), {1, 2, 3, 4}, 1e-6f));
}

TEST(vec4, neg)
{
    EXPECT_TRUE(i3_vec4_eq(i3_vec4_neg({1, -2, 3, -4}), {-1, 2, -3, 4}, 1e-6f));
}

TEST(vec4, add)
{
    EXPECT_TRUE(i3_vec4_eq(i3_vec4_add({1, 2, 3, 4}, {5, 6, 7, 8}), {6, 8, 10, 12}, 1e-6f));
}

TEST(vec4, sub)
{
    EXPECT_TRUE(i3_vec4_eq(i3_vec4_sub({1, 2, 3, 4}, {5, 6, 7, 8}), {-4, -4, -4, -4}, 1e-6f));
}

TEST(vec4, mul)
{
    EXPECT_TRUE(i3_vec4_eq(i3_vec4_mul({1, 2, 3, 4}, {5, 6, 7, 8}), {5, 12, 21, 32}, 1e-6f));
}

TEST(vec4, div)
{
    EXPECT_TRUE(i3_vec4_eq(i3_vec4_div({1, 2, 3, 4}, {5, 6, 7, 8}), {0.2f, 0.333333f, 0.428571f, 0.5f}, 1e-6f));
}

TEST(vec4, scale)
{
    EXPECT_TRUE(i3_vec4_eq(i3_vec4_scale({1, 2, 3, 4}, 2), {2, 4, 6, 8}, 1e-6f));
}

TEST(vec4, dot)
{
    EXPECT_FLOAT_EQ(i3_vec4_dot({1, 2, 3, 4}, {5, 6, 7, 8}), 70.0f);
}

TEST(vec4, len2)
{
    EXPECT_FLOAT_EQ(i3_vec4_len2({3, 4, 5, 6}), 86.0f);
}

TEST(vec4, len)
{
    EXPECT_FLOAT_EQ(i3_vec4_len({3, 4, 5, 6}), 9.273618495f);
}

TEST(vec4, normalize)
{
    EXPECT_TRUE(i3_vec4_eq(i3_vec4_normalize({3, 4, 5, 6}), {0.323498f, 0.431331f, 0.539164f, 0.646997f}, 1e-6f));
}

TEST(vec4, min)
{
    EXPECT_TRUE(i3_vec4_eq(i3_vec4_min({1, 2, 3, 4}, {5, 6, 7, 8}), {1, 2, 3, 4}, 1e-6f));
}

TEST(vec4, max)
{
    EXPECT_TRUE(i3_vec4_eq(i3_vec4_max({1, 2, 3, 4}, {5, 6, 7, 8}), {5, 6, 7, 8}, 1e-6f));
}

TEST(vec4, clamp)
{
    EXPECT_TRUE(i3_vec4_eq(i3_vec4_clamp({1, 2, 3, 4}, {0, 0, 0, 0}, {5, 5, 5, 5}), {1, 2, 3, 4}, 1e-6f));
    EXPECT_TRUE(i3_vec4_eq(i3_vec4_clamp({-1, -2, -3, -4}, {0, 0, 0, 0}, {5, 5, 5, 5}), {0, 0, 0, 0}, 1e-6f));
    EXPECT_TRUE(i3_vec4_eq(i3_vec4_clamp({6, 7, 8, 9}, {0, 0, 0, 0}, {5, 5, 5, 5}), {5, 5, 5, 5}, 1e-6f));
}

TEST(vec4, saturate)
{
    EXPECT_TRUE(i3_vec4_eq(i3_vec4_saturate({-1, 2, 3, 4}), {0, 1, 1, 1}, 1e-6f));
    EXPECT_TRUE(i3_vec4_eq(i3_vec4_saturate({6, 7, 8, 9}), {1, 1, 1, 1}, 1e-6f));
}
