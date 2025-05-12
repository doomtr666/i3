#pragma once

#include "mat.h"
#include "quat.h"

static inline i3_mat4_t i3_mat4_translation(i3_vec3_t translation);
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