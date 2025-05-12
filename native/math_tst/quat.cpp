#include <gtest/gtest.h>

extern "C"
{
#include "native/math/quat.h"
}

TEST(quat, rotation_xyz)
{
    // rotation around x axis by 90 degrees
    i3_quat_t qx = i3_quat_axis_angle(i3_vec3(1.0f, 0.0f, 0.0f), I3_PI_OVER_2);
    i3_vec3_t vx = i3_vec3(0.0f, 1.0f, 0.0f);
    i3_vec3_t rx = i3_quat_transform(qx, vx);
    EXPECT_TRUE(i3_vec3_eq(rx, i3_vec3(0.0f, 0.0f, 1.0f), 1e-6f));

    // rotation around y axis by 90 degrees
    i3_quat_t qy = i3_quat_axis_angle(i3_vec3(0.0f, 1.0f, 0.0f), I3_PI_OVER_2);
    i3_vec3_t vy = i3_vec3(0.0f, 0.0f, 1.0f);
    i3_vec3_t ry = i3_quat_transform(qy, vy);
    EXPECT_TRUE(i3_vec3_eq(ry, i3_vec3(1.0f, 0.0f, 0.0f), 1e-6f));

    // rotation around z axis by 90 degrees
    i3_quat_t qz = i3_quat_axis_angle(i3_vec3(0.0f, 0.0f, 1.0f), I3_PI_OVER_2);
    i3_vec3_t vz = i3_vec3(1.0f, 0.0f, 0.0f);
    i3_vec3_t rz = i3_quat_transform(qz, vz);
    EXPECT_TRUE(i3_vec3_eq(rz, i3_vec3(0.0f, 1.0f, 0.0f), 1e-6f));
}