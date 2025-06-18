#pragma once

#include "mat.h"
#include "quat.h"

static inline i3_mat4_t i3_mat4_translation(i3_vec3_t translation);
static inline i3_mat4_t i3_mat4_rotation_basis(i3_vec3_t right, i3_vec3_t up, i3_vec3_t forward);
static inline i3_mat4_t i3_mat4_rotation_x(float angle);
static inline i3_mat4_t i3_mat4_rotation_y(float angle);
static inline i3_mat4_t i3_mat4_rotation_z(float angle);
static inline i3_mat4_t i3_mat4_rotation_euler(i3_vec3_t angles);  // pitch, yaw, roll
static inline i3_mat4_t i3_mat4_rotation_axis(i3_vec3_t axis, float angle);
static inline i3_mat4_t i3_mat4_rotation_quat(i3_quat_t q);
static inline i3_mat4_t i3_mat4_scale(i3_vec3_t scale);
static inline i3_mat4_t i3_mat4_persective_fov_rh(float fov, float aspect, float near, float far);
static inline i3_mat4_t i3_mat4_look_to_rh(i3_vec3_t position, i3_vec3_t direction, i3_vec3_t up);
static inline i3_mat4_t i3_mat4_look_at_rh(i3_vec3_t position, i3_vec3_t target, i3_vec3_t up);

// implementation

static inline i3_mat4_t i3_mat4_translation(i3_vec3_t translation)
{
    i3_mat4_t m;
    m.m00 = 1.0f;
    m.m10 = 0.0f;
    m.m20 = 0.0f;
    m.m30 = 0.0f;
    m.m01 = 0.0f;
    m.m11 = 1.0f;
    m.m21 = 0.0f;
    m.m31 = 0.0f;
    m.m02 = 0.0f;
    m.m12 = 0.0f;
    m.m22 = 1.0f;
    m.m32 = 0.0f;
    m.m03 = translation.x;
    m.m13 = translation.y;
    m.m23 = translation.z;
    m.m33 = 1.0f;

    return m;
}

static inline i3_mat4_t i3_mat4_rotation_basis(i3_vec3_t right, i3_vec3_t up, i3_vec3_t forward)
{
    i3_mat4_t m;

    m.m00 = right.x;
    m.m10 = right.y;
    m.m20 = right.z;
    m.m30 = 0.0f;
    m.m01 = up.x;
    m.m11 = up.y;
    m.m21 = up.z;
    m.m31 = 0.0f;
    m.m02 = forward.x;
    m.m12 = forward.y;
    m.m22 = forward.z;
    m.m32 = 0.0f;
    m.m03 = 0.0f;
    m.m13 = 0.0f;
    m.m23 = 0.0f;
    m.m33 = 1.0f;

    return m;
}

static inline i3_mat4_t i3_mat4_rotation_x(float angle)
{
    float s = i3_sinf(angle);
    float c = i3_cosf(angle);

    i3_mat4_t m;
    m.m00 = 1.0f;
    m.m10 = 0.0f;
    m.m20 = 0.0f;
    m.m30 = 0.0f;
    m.m01 = 0.0f;
    m.m11 = c;
    m.m21 = s;
    m.m31 = 0.0f;
    m.m02 = 0.0f;
    m.m12 = -s;
    m.m22 = c;
    m.m32 = 0.0f;
    m.m03 = 0.0f;
    m.m13 = 0.0f;
    m.m23 = 0.0f;
    m.m33 = 1.0f;
    return m;
}

static inline i3_mat4_t i3_mat4_rotation_y(float angle)
{
    float s = i3_sinf(angle);
    float c = i3_cosf(angle);

    i3_mat4_t m;
    m.m00 = c;
    m.m10 = 0.0f;
    m.m20 = -s;
    m.m30 = 0.0f;
    m.m01 = 0.0f;
    m.m11 = 1.0f;
    m.m21 = 0.0f;
    m.m31 = 0.0f;
    m.m02 = s;
    m.m12 = 0.0f;
    m.m22 = c;
    m.m32 = 0.0f;
    m.m03 = 0.0f;
    m.m13 = 0.0f;
    m.m23 = 0.0f;
    m.m33 = 1.0f;
    return m;
}

static inline i3_mat4_t i3_mat4_rotation_z(float angle)
{
    float s = i3_sinf(angle);
    float c = i3_cosf(angle);

    i3_mat4_t m;
    m.m00 = c;
    m.m10 = s;
    m.m20 = 0.0f;
    m.m30 = 0.0f;
    m.m01 = -s;
    m.m11 = c;
    m.m21 = 0.0f;
    m.m31 = 0.0f;
    m.m02 = 0.0f;
    m.m12 = 0.0f;
    m.m22 = 1.0f;
    m.m32 = 0.0f;
    m.m03 = 0.0f;
    m.m13 = 0.0f;
    m.m23 = 0.0f;
    m.m33 = 1.0f;
    return m;
}

static inline i3_mat4_t i3_mat4_rotation_euler(i3_vec3_t angles)
{
    float cp = i3_cosf(angles.x);
    float sp = i3_sinf(angles.x);

    float cy = i3_cosf(angles.y);
    float sy = i3_sinf(angles.y);

    float cr = i3_cosf(angles.z);
    float sr = i3_sinf(angles.z);

    i3_mat4_t m;
    m.m00 = cr * cy + sr * sp * sy;
    m.m10 = sr * cp;
    m.m20 = sr * sp * cy - cr * sy;
    m.m30 = 0.0f;
    m.m01 = cr * sp * sy - sr * cy;
    m.m11 = cr * cp;
    m.m21 = sr * sy + cr * sp * cy;
    m.m31 = 0.0f;
    m.m02 = cp * sy;
    m.m12 = -sp;
    m.m22 = cp * cy;
    m.m32 = 0.0f;
    m.m03 = 0.0f;
    m.m13 = 0.0f;
    m.m23 = 0.0f;
    m.m33 = 1.0f;
    return m;
}

static inline i3_mat4_t i3_mat4_rotation_axis(i3_vec3_t axis, float angle)
{
    return i3_mat4_rotation_quat(i3_quat_axis_angle(axis, angle));
}

static inline i3_mat4_t i3_mat4_rotation_quat(i3_quat_t q)
{
    float qxx = q.b * q.b;
    float qyy = q.c * q.c;
    float qzz = q.d * q.d;

    i3_mat4_t r;
    r.m00 = 1.0f - 2.0f * (qyy + qzz);
    r.m10 = 2.0f * (q.b * q.c + q.d * q.a);
    r.m20 = 2.0f * (q.b * q.d - q.c * q.a);
    r.m30 = 0.0f;
    r.m01 = 2.0f * (q.b * q.c - q.d * q.a);
    r.m11 = 1.0f - 2.0f * (qxx + qzz);
    r.m21 = 2.0f * (q.c * q.d + q.b * q.a);
    r.m31 = 0.0f;
    r.m02 = 2.0f * (q.b * q.d + q.c * q.a);
    r.m12 = 2.0f * (q.c * q.d - q.b * q.a);
    r.m22 = 1.0f - 2.0f * (qxx + qyy);
    r.m32 = 0.0f;
    r.m03 = 0.0f;
    r.m13 = 0.0f;
    r.m23 = 0.0f;
    r.m33 = 1.0f;
    return r;
}

static inline i3_mat4_t i3_mat4_scale(i3_vec3_t scale)
{
    i3_mat4_t m;
    m.m00 = scale.x;
    m.m10 = 0.0f;
    m.m20 = 0.0f;
    m.m30 = 0.0f;
    m.m01 = 0.0f;
    m.m11 = scale.y;
    m.m21 = 0.0f;
    m.m31 = 0.0f;
    m.m02 = 0.0f;
    m.m12 = 0.0f;
    m.m22 = scale.z;
    m.m32 = 0.0f;
    m.m03 = 0.0f;
    m.m13 = 0.0f;
    m.m23 = 0.0f;
    m.m33 = 1.0f;
    return m;
}

static inline i3_mat4_t i3_mat4_persective_fov_rh(float fov, float aspect, float near, float far)
{
    float height = 1.0f / i3_tanf(fov * 0.5f);
    float width = height / aspect;
    float range = far / (near - far);

    i3_mat4_t m;
    m.m00 = width;
    m.m10 = 0.0f;
    m.m20 = 0.0f;
    m.m30 = 0.0f;
    m.m01 = 0.0f;
    m.m11 = height;
    m.m21 = 0.0f;
    m.m31 = 0.0f;
    m.m02 = 0.0f;
    m.m12 = 0.0f;
    m.m22 = range;
    m.m32 = -1.0f;
    m.m03 = 0.0f;
    m.m13 = 0.0f;
    m.m23 = range * near;
    m.m33 = 0.0f;
    return m;
}

static inline i3_mat4_t i3_mat4_look_to_rh(i3_vec3_t position, i3_vec3_t direction, i3_vec3_t up)
{
    // basis
    i3_vec3_t forward_vec = i3_vec3_normalize(direction);
    i3_vec3_t right_vec = i3_vec3_normalize(i3_vec3_cross(up, forward_vec));
    i3_vec3_t up_vec = i3_vec3_cross(forward_vec, right_vec);

    // translation
    i3_vec3_t t = i3_vec3_neg(position);

    // TODO: check if this is correctly unfolded by the compiler, may need to be reduced manually
    return i3_mat4_mult(i3_mat4_rotation_basis(right_vec, up_vec, forward_vec), i3_mat4_translation(t));
}

static inline i3_mat4_t i3_mat4_look_at_rh(i3_vec3_t position, i3_vec3_t target, i3_vec3_t up)
{
    return i3_mat4_look_to_rh(position, i3_vec3_sub(target, position), up);
}
