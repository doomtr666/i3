#pragma once

#include "mat.h"
#include "quat.h"

static inline i3_mat4_t i3_mat4_translation(i3_vec3_t translation);
static inline i3_mat4_t i3_mat4_rotation_x(float angle);
static inline i3_mat4_t i3_mat4_rotation_y(float angle);
static inline i3_mat4_t i3_mat4_rotation_z(float angle);
static inline i3_mat4_t i3_mat4_rotation_euler(i3_vec3_t angles);  // pitch, yaw, roll
static inline i3_mat4_t i3_mat4_rotation_axis(i3_vec3_t axis, float angle);
static inline i3_mat4_t i3_mat4_rotation_quat(i3_quat_t q);
static inline i3_mat4_t i3_mat4_scale(i3_vec3_t scale);
static inline i3_mat4_t i3_mat4_persective_fov_rh(float fov, float aspect, float near, float far);
static inline i3_mat4_t i3_mat4_persective_fov_lh(float fov, float aspect, float near, float far);

// implementation

static inline i3_mat4_t i3_mat4_translation(i3_vec3_t translation)
{
    i3_mat4_t m;
    m.m00 = 1.0f;
    m.m01 = 0.0f;
    m.m02 = 0.0f;
    m.m03 = 0.0f;

    m.m10 = 0.0f;
    m.m11 = 1.0f;
    m.m12 = 0.0f;
    m.m13 = 0.0f;

    m.m20 = 0.0f;
    m.m21 = 0.0f;
    m.m22 = 1.0f;
    m.m23 = 0.0f;

    m.m30 = translation.x;
    m.m31 = translation.y;
    m.m32 = translation.z;
    m.m33 = 1.0f;

    return m;
}

static inline i3_mat4_t i3_mat4_rotation_x(float angle)
{
    float s = i3_sinf(angle);
    float c = i3_cosf(angle);

    i3_mat4_t m;
    m.m00 = 1.0f;
    m.m01 = 0.0f;
    m.m02 = 0.0f;
    m.m03 = 0.0f;

    m.m10 = 0.0f;
    m.m11 = c;
    m.m12 = s;
    m.m13 = 0.0f;

    m.m20 = 0.0f;
    m.m21 = -s;
    m.m22 = c;

    m.m23 = 0.0f;
    m.m30 = 0.0f;
    m.m31 = 0.0f;
    m.m32 = 0.0f;
    m.m33 = 1.0f;

    return m;
}

static inline i3_mat4_t i3_mat4_rotation_y(float angle)
{
    float s = i3_sinf(angle);
    float c = i3_cosf(angle);

    i3_mat4_t m;
    m.m00 = c;
    m.m01 = 0.0f;
    m.m02 = -s;
    m.m03 = 0.0f;

    m.m10 = 0.0f;
    m.m11 = 1.0f;
    m.m12 = 0.0f;
    m.m13 = 0.0f;

    m.m20 = s;
    m.m21 = 0.0f;
    m.m22 = c;
    m.m23 = 0.0f;

    m.m30 = 0.0f;
    m.m31 = 0.0f;
    m.m32 = 0.0f;
    m.m33 = 1.0f;

    return m;
}

static inline i3_mat4_t i3_mat4_rotation_z(float angle)
{
    float s = i3_sinf(angle);
    float c = i3_cosf(angle);

    i3_mat4_t m;
    m.m00 = c;
    m.m01 = s;
    m.m02 = 0.0f;
    m.m03 = 0.0f;

    m.m10 = -s;
    m.m11 = c;
    m.m12 = 0.0f;
    m.m13 = 0.0f;

    m.m20 = 0.0f;
    m.m21 = 0.0f;
    m.m22 = 1.0f;
    m.m23 = 0.0f;

    m.m30 = 0.0f;
    m.m31 = 0.0f;
    m.m32 = 0.0f;
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
    m.m01 = sr * cp;
    m.m02 = sr * sp * cy - cr * sy;
    m.m03 = 0.0f;

    m.m10 = cr * sp * sy - sr * cy;
    m.m11 = cr * cp;
    m.m12 = sr * sy + cr * sp * cy;
    m.m13 = 0.0f;

    m.m20 = cp * sy;
    m.m21 = -sp;
    m.m22 = cp * cy;
    m.m23 = 0.0f;
    m.m30 = 0.0f;
    m.m31 = 0.0f;
    m.m32 = 0.0f;
    m.m33 = 1.0f;

    return m;
}

static inline i3_mat4_t i3_mat4_rotation_axis(i3_vec3_t axis, float angle)
{
    return i3_mat4_rotation_quat(i3_quat_axis_angle(axis, angle));
}

static inline i3_mat4_t i3_mat4_rotation_quat(i3_quat_t q)
{
    // TODO
}

static inline i3_mat4_t i3_mat4_scale(i3_vec3_t scale)
{
    i3_mat4_t m;
    m.m00 = scale.x;
    m.m01 = 0.0f;
    m.m02 = 0.0f;
    m.m03 = 0.0f;

    m.m10 = 0.0f;
    m.m11 = scale.y;
    m.m12 = 0.0f;
    m.m13 = 0.0f;

    m.m20 = 0.0f;
    m.m21 = 0.0f;
    m.m22 = scale.z;
    m.m23 = 0.0f;

    m.m30 = 0.0f;
    m.m31 = 0.0f;
    m.m32 = 0.0f;
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
    m.m01 = 0.0f;
    m.m02 = 0.0f;
    m.m03 = 0.0f;

    m.m10 = 0.0f;
    m.m11 = height;
    m.m12 = 0.0f;
    m.m13 = 0.0f;

    m.m20 = 0.0f;
    m.m21 = 0.0f;
    m.m22 = range;
    m.m23 = -1.0f;

    m.m30 = 0.0f;
    m.m31 = 0.0f;
    m.m32 = range * near;
    m.m33 = 0.0f;

    return m;
}

static inline i3_mat4_t i3_mat4_persective_fov_lh(float fov, float aspect, float near, float far)
{
    float height = 1.0f / i3_tanf(fov * 0.5f);
    float width = height / aspect;
    float range = far / (far - near);

    i3_mat4_t m;
    m.m00 = width;
    m.m01 = 0.0f;
    m.m02 = 0.0f;
    m.m03 = 0.0f;

    m.m10 = 0.0f;
    m.m11 = height;
    m.m12 = 0.0f;
    m.m13 = 0.0f;

    m.m20 = 0.0f;
    m.m21 = 0.0f;
    m.m22 = range;
    m.m23 = 1.0f;

    m.m30 = 0.0f;
    m.m31 = 0.0f;
    m.m32 = -range * near;
    m.m33 = 0.0f;

    return m;
}