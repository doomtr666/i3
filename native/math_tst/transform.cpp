#include <gtest/gtest.h>

extern "C"
{
#include "native/math/transform.h"
}

TEST(transform, translation)
{
    i3_vec3_t translation = {1.0f, 2.0f, 3.0f};
    i3_mat4_t m = i3_mat4_translation(translation);

    i3_vec4_t x = {5.0f, 6.0f, 7.0f, 1.0f};
    i3_vec4_t tx = i3_mat4_mult_vec4(m, x);

    EXPECT_FLOAT_EQ(tx.x, x.x + translation.x);
    EXPECT_FLOAT_EQ(tx.y, x.y + translation.y);
    EXPECT_FLOAT_EQ(tx.z, x.z + translation.z);
    EXPECT_FLOAT_EQ(tx.w, x.w);
}

TEST(transform, look_to)
{
    i3_vec3_t position = {0.0f, 0.0f, 0.0f};
    i3_vec3_t direction = {0.0f, 0.0f, 1.0f};
    i3_vec3_t up = {0.0f, 1.0f, 0.0f};

    i3_mat4_t m = i3_mat4_look_to_rh(position, direction, up);

    i3_vec4_t x = {1.0f, 2.0f, 3.0f, 1.0f};
    i3_vec4_t tx = i3_mat4_mult_vec4(m, x);

    EXPECT_FLOAT_EQ(tx.x, x.x);
    EXPECT_FLOAT_EQ(tx.y, x.y);
    EXPECT_FLOAT_EQ(tx.z, -x.z);
    EXPECT_FLOAT_EQ(tx.w, x.w);
}