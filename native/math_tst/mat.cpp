#include <gtest/gtest.h>

extern "C"
{
#include "native/math/mat.h"
}

TEST(mat22, set)
{
    EXPECT_TRUE(i3_mat22_eq(i3_mat22_set(1.0f), {1.0f, 1.0f, 1.0f, 1.0f}, 1e-6f));
}

TEST(mat34, transpose)
{
    i3_mat34_t a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    i3_mat43_t b = {1.0f, 5.0f, 9.0f, 2.0f, 6.0f, 10.0f, 3.0f, 7.0f, 11.0f, 4.0f, 8.0f, 12.0f};
    EXPECT_TRUE(i3_mat43_eq(i3_mat34_transpose(a), b, 1e-6f));
}

TEST(mat33, indentity)
{
    i3_mat33_t a = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    EXPECT_TRUE(i3_mat33_eq(i3_mat33_identity(), a, 1e-6f));
}