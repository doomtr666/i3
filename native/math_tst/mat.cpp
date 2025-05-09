#include <gtest/gtest.h>

extern "C"
{
#include "native/math/mat.h"
}

// mat2

TEST(mat2, det)
{
    i3_mat2_t m = i3_mat2(4, 1, 8, 7);
    float det = i3_mat2_det(m);
    EXPECT_TRUE(i3_eqf(det, 20.0f, 1e-6f));
}

TEST(mat2, inverse)
{
    i3_mat2_t mat = i3_mat2(4, 1, 8, 7);
    i3_mat2_t inv = i3_mat2_invert(mat);
    i3_mat2_t r1 = i3_mat2_mult(mat, inv);
    i3_mat2_t r2 = i3_mat2_mult(inv, mat);
    EXPECT_TRUE(i3_mat2_eq(r1, i3_mat2_identity(), 1e-6f));
    EXPECT_TRUE(i3_mat2_eq(r2, i3_mat2_identity(), 1e-6f));
}

// mat3

TEST(mat3, det)
{
    i3_mat3_t m = i3_mat3(4, 1, 8, 7, 8, 3, 1, 5, 9);
    float det = i3_mat3_det(m);
    EXPECT_TRUE(i3_eqf(det, 384.0f, 1e-6f));
}

TEST(mat3, inverse)
{
    i3_mat3_t mat = i3_mat3(4, 1, 8, 7, 8, 3, 1, 5, 9);
    i3_mat3_t inv = i3_mat3_invert(mat);
    i3_mat3_t r1 = i3_mat3_mult(mat, inv);
    i3_mat3_t r2 = i3_mat3_mult(inv, mat);
    EXPECT_TRUE(i3_mat3_eq(r1, i3_mat3_identity(), 1e-6f));
    EXPECT_TRUE(i3_mat3_eq(r2, i3_mat3_identity(), 1e-6f));
}

// mat4

TEST(mat4, det)
{
    i3_mat4_t m = i3_mat4(4, 1, 3, 7, 8, 7, 6, 5, 0, 2, 4, 6, 3, 2, 5, 1);
    float det = i3_mat4_det(m);
    EXPECT_TRUE(i3_eqf(det, -688.0f, 1e-6f));
}

TEST(mat4, inverse)
{
    i3_mat4_t mat = i3_mat4(4, 1, 3, 7, 8, 7, 6, 5, 0, 2, 4, 6, 3, 2, 5, 1);
    i3_mat4_t inv = i3_mat4_invert(mat);

    i3_mat4_t r1 = i3_mat4_mult(mat, inv);
    i3_mat4_t r2 = i3_mat4_mult(inv, mat);
    EXPECT_TRUE(i3_mat4_eq(r1, i3_mat4_identity(), 1e-6f));
    EXPECT_TRUE(i3_mat4_eq(r2, i3_mat4_identity(), 1e-6f));
}
