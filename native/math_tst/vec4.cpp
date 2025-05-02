#include <gtest/gtest.h>

extern "C"
{
#include "native/math/vec.h"
}

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
    i3_vec4_t vec = {3, 4, 5, 6};
    i3_vec4_t norm = i3_vec4_normalize(vec);

    std::cout << i3_vec4_dump(norm) << std::endl;

    // len = sqrt(3^2 + 4^2 + 5^2 + 6^2) = sqrt(86) = 9.273618495
    // normalize = {3, 4, 5, 6} / 9.273618495 = {0.323607f, 0.430143f, 0.537679f, 0.645215f}
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
    EXPECT_TRUE(i3_vec4_eq(i3_vec4_clamp({1, 2, 3, 4}, 0, 5), {1, 2, 3, 4}, 1e-6f));
    EXPECT_TRUE(i3_vec4_eq(i3_vec4_clamp({-1, -2, -3, -4}, 0, 5), {0, 0, 0, 0}, 1e-6f));
    EXPECT_TRUE(i3_vec4_eq(i3_vec4_clamp({6, 7, 8, 9}, 0, 5), {5, 5, 5, 5}, 1e-6f));
}

TEST(vec4, saturate)
{
    EXPECT_TRUE(i3_vec4_eq(i3_vec4_saturate({-1, 2, 3, 4}), {0, 1, 1, 1}, 1e-6f));
    EXPECT_TRUE(i3_vec4_eq(i3_vec4_saturate({6, 7, 8, 9}), {1, 1, 1, 1}, 1e-6f));
}
