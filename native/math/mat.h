#pragma once
// This file is generated by mathlib_generator, do not edit manually.

#include "vec.h"

// i3_mat2_t
typedef struct i3_mat2_t
{
    union
    {
        float m[4];
        struct
        {
            float m00, m01;
            float m10, m11;
        };
        struct
        {
            i3_vec2_t v0;
            i3_vec2_t v1;
        };
    };
} i3_mat2_t;

// i3_mat3_t
typedef struct i3_mat3_t
{
    union
    {
        float m[9];
        struct
        {
            float m00, m01, m02;
            float m10, m11, m12;
            float m20, m21, m22;
        };
        struct
        {
            i3_vec3_t v0;
            i3_vec3_t v1;
            i3_vec3_t v2;
        };
    };
} i3_mat3_t;

// i3_mat34_t
typedef struct i3_mat34_t
{
    union
    {
        float m[12];
        struct
        {
            float m00, m01, m02, m03;
            float m10, m11, m12, m13;
            float m20, m21, m22, m23;
        };
        struct
        {
            i3_vec4_t v0;
            i3_vec4_t v1;
            i3_vec4_t v2;
        };
    };
} i3_mat34_t;

// i3_mat43_t
typedef struct i3_mat43_t
{
    union
    {
        float m[12];
        struct
        {
            float m00, m01, m02;
            float m10, m11, m12;
            float m20, m21, m22;
            float m30, m31, m32;
        };
        struct
        {
            i3_vec3_t v0;
            i3_vec3_t v1;
            i3_vec3_t v2;
            i3_vec3_t v3;
        };
    };
} i3_mat43_t;

// i3_mat4_t
typedef struct i3_mat4_t
{
    union
    {
        float m[16];
        struct
        {
            float m00, m01, m02, m03;
            float m10, m11, m12, m13;
            float m20, m21, m22, m23;
            float m30, m31, m32, m33;
        };
        struct
        {
            i3_vec4_t v0;
            i3_vec4_t v1;
            i3_vec4_t v2;
            i3_vec4_t v3;
        };
    };
} i3_mat4_t;

// i3_mat2_t
static inline i3_mat2_t i3_mat2(float m00, float m01, float m10, float m11);
static inline i3_mat2_t i3_mat2_set(float v);
static inline i3_mat2_t i3_mat2_zero();
static inline i3_mat2_t i3_mat2_identity();
static inline i3_mat2_t i3_mat2_transpose(i3_mat2_t m);
static inline i3_mat2_t i3_mat2_mult(i3_mat2_t a, i3_mat2_t b);
static inline i3_vec2_t i3_mat2_mult_vec2(i3_mat2_t m, i3_vec2_t v);
static inline i3_vec2_t i3_vec2_mult_mat2(i3_vec2_t v, i3_mat2_t m);
static inline float i3_mat2_det(i3_mat2_t m);
static inline i3_mat2_t i3_mat2_invert(i3_mat2_t m);
static inline bool i3_mat2_eq(i3_mat2_t a, i3_mat2_t b, float epsilon);
static inline bool i3_mat2_neq(i3_mat2_t a, i3_mat2_t b, float epsilon);
static inline const char* i3_mat2_str(i3_mat2_t m);

// i3_mat3_t
static inline i3_mat3_t
i3_mat3(float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22);
static inline i3_mat3_t i3_mat3_set(float v);
static inline i3_mat3_t i3_mat3_zero();
static inline i3_mat3_t i3_mat3_identity();
static inline i3_mat3_t i3_mat3_transpose(i3_mat3_t m);
static inline i3_mat3_t i3_mat3_mult(i3_mat3_t a, i3_mat3_t b);
static inline i3_mat34_t i3_mat3_mult_mat34(i3_mat3_t a, i3_mat34_t b);
static inline i3_vec3_t i3_mat3_mult_vec3(i3_mat3_t m, i3_vec3_t v);
static inline i3_vec3_t i3_vec3_mult_mat3(i3_vec3_t v, i3_mat3_t m);
static inline i3_mat2_t i3_mat3_submat(i3_mat3_t m, int row, int col);
static inline float i3_mat3_minor(i3_mat3_t m, int row, int col);
static inline float i3_mat3_det(i3_mat3_t m);
static inline i3_mat3_t i3_mat3_invert(i3_mat3_t m);
static inline bool i3_mat3_eq(i3_mat3_t a, i3_mat3_t b, float epsilon);
static inline bool i3_mat3_neq(i3_mat3_t a, i3_mat3_t b, float epsilon);
static inline const char* i3_mat3_str(i3_mat3_t m);

// i3_mat34_t
static inline i3_mat34_t i3_mat34(float m00,
                                  float m01,
                                  float m02,
                                  float m03,
                                  float m10,
                                  float m11,
                                  float m12,
                                  float m13,
                                  float m20,
                                  float m21,
                                  float m22,
                                  float m23);
static inline i3_mat34_t i3_mat34_set(float v);
static inline i3_mat34_t i3_mat34_zero();
static inline i3_mat43_t i3_mat34_transpose(i3_mat34_t m);
static inline i3_mat3_t i3_mat34_mult_mat43(i3_mat34_t a, i3_mat43_t b);
static inline i3_mat34_t i3_mat34_mult_mat4(i3_mat34_t a, i3_mat4_t b);
static inline i3_vec3_t i3_mat34_mult_vec4(i3_mat34_t m, i3_vec4_t v);
static inline i3_vec4_t i3_vec3_mult_mat34(i3_vec3_t v, i3_mat34_t m);
static inline bool i3_mat34_eq(i3_mat34_t a, i3_mat34_t b, float epsilon);
static inline bool i3_mat34_neq(i3_mat34_t a, i3_mat34_t b, float epsilon);
static inline const char* i3_mat34_str(i3_mat34_t m);

// i3_mat43_t
static inline i3_mat43_t i3_mat43(float m00,
                                  float m01,
                                  float m02,
                                  float m10,
                                  float m11,
                                  float m12,
                                  float m20,
                                  float m21,
                                  float m22,
                                  float m30,
                                  float m31,
                                  float m32);
static inline i3_mat43_t i3_mat43_set(float v);
static inline i3_mat43_t i3_mat43_zero();
static inline i3_mat34_t i3_mat43_transpose(i3_mat43_t m);
static inline i3_mat43_t i3_mat43_mult_mat3(i3_mat43_t a, i3_mat3_t b);
static inline i3_mat4_t i3_mat43_mult_mat34(i3_mat43_t a, i3_mat34_t b);
static inline i3_vec4_t i3_mat43_mult_vec3(i3_mat43_t m, i3_vec3_t v);
static inline i3_vec3_t i3_vec4_mult_mat43(i3_vec4_t v, i3_mat43_t m);
static inline bool i3_mat43_eq(i3_mat43_t a, i3_mat43_t b, float epsilon);
static inline bool i3_mat43_neq(i3_mat43_t a, i3_mat43_t b, float epsilon);
static inline const char* i3_mat43_str(i3_mat43_t m);

// i3_mat4_t
static inline i3_mat4_t i3_mat4(float m00,
                                float m01,
                                float m02,
                                float m03,
                                float m10,
                                float m11,
                                float m12,
                                float m13,
                                float m20,
                                float m21,
                                float m22,
                                float m23,
                                float m30,
                                float m31,
                                float m32,
                                float m33);
static inline i3_mat4_t i3_mat4_set(float v);
static inline i3_mat4_t i3_mat4_zero();
static inline i3_mat4_t i3_mat4_identity();
static inline i3_mat4_t i3_mat4_transpose(i3_mat4_t m);
static inline i3_mat43_t i3_mat4_mult_mat43(i3_mat4_t a, i3_mat43_t b);
static inline i3_mat4_t i3_mat4_mult(i3_mat4_t a, i3_mat4_t b);
static inline i3_vec4_t i3_mat4_mult_vec4(i3_mat4_t m, i3_vec4_t v);
static inline i3_vec4_t i3_vec4_mult_mat4(i3_vec4_t v, i3_mat4_t m);
static inline i3_mat3_t i3_mat4_submat(i3_mat4_t m, int row, int col);
static inline float i3_mat4_minor(i3_mat4_t m, int row, int col);
static inline float i3_mat4_det(i3_mat4_t m);
static inline i3_mat4_t i3_mat4_invert(i3_mat4_t m);
static inline bool i3_mat4_eq(i3_mat4_t a, i3_mat4_t b, float epsilon);
static inline bool i3_mat4_neq(i3_mat4_t a, i3_mat4_t b, float epsilon);
static inline const char* i3_mat4_str(i3_mat4_t m);

// implementation of i3_mat2_t
static inline i3_mat2_t i3_mat2(float m00, float m01, float m10, float m11)
{
    i3_mat2_t r;
    r.m00 = m00;
    r.m01 = m01;
    r.m10 = m10;
    r.m11 = m11;
    return r;
}

static inline i3_mat2_t i3_mat2_set(float v)
{
    i3_mat2_t r;
    r.m00 = v;
    r.m01 = v;
    r.m10 = v;
    r.m11 = v;
    return r;
}

static inline i3_mat2_t i3_mat2_zero()
{
    return i3_mat2_set(0);
}

static inline i3_mat2_t i3_mat2_identity()
{
    i3_mat2_t r;
    r.m00 = 1;
    r.m01 = 0;
    r.m10 = 0;
    r.m11 = 1;
    return r;
}

static inline i3_mat2_t i3_mat2_transpose(i3_mat2_t m)
{
    i3_mat2_t r;
    r.m00 = m.m00;
    r.m01 = m.m10;
    r.m10 = m.m01;
    r.m11 = m.m11;
    return r;
}

static inline i3_mat2_t i3_mat2_mult(i3_mat2_t a, i3_mat2_t b)
{
    i3_mat2_t r;
    r.m00 = a.m00 * b.m00 + a.m01 * b.m10;
    r.m01 = a.m00 * b.m01 + a.m01 * b.m11;
    r.m10 = a.m10 * b.m00 + a.m11 * b.m10;
    r.m11 = a.m10 * b.m01 + a.m11 * b.m11;
    return r;
}

static inline i3_vec2_t i3_mat2_mult_vec2(i3_mat2_t m, i3_vec2_t v)
{
    i3_vec2_t r;
    r.x = m.m00 * v.x + m.m01 * v.y;
    r.y = m.m10 * v.x + m.m11 * v.y;
    return r;
}

static inline i3_vec2_t i3_vec2_mult_mat2(i3_vec2_t v, i3_mat2_t m)
{
    i3_vec2_t r;
    r.x = v.x * m.m00 + v.y * m.m10;
    r.y = v.x * m.m01 + v.y * m.m11;
    return r;
}

static inline float i3_mat2_det(i3_mat2_t m)
{
    return m.m00 * m.m11 - m.m01 * m.m10;
}

static inline i3_mat2_t i3_mat2_invert(i3_mat2_t m)
{
    float det = i3_mat2_det(m);
    i3_mat2_t r;
    r.m00 = m.m11 / det;
    r.m01 = -m.m01 / det;
    r.m10 = -m.m10 / det;
    r.m11 = m.m00 / det;
    return r;
}

static inline bool i3_mat2_eq(i3_mat2_t a, i3_mat2_t b, float epsilon)
{
    for (int i = 0; i < 4; ++i)
        if (!i3_eqf(a.m[i], b.m[i], epsilon))
            return false;
    return true;
}

static inline bool i3_mat2_neq(i3_mat2_t a, i3_mat2_t b, float epsilon)
{
    return !i3_mat2_eq(a, b, epsilon);
}

static inline const char* i3_mat2_str(i3_mat2_t m)
{
    static char buffer[64];
    snprintf(buffer, sizeof(buffer), "{{%f, %f}, {%f, %f}}", m.m00, m.m01, m.m10, m.m11);
    return buffer;
}

// implementation of i3_mat3_t
static inline i3_mat3_t
i3_mat3(float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22)
{
    i3_mat3_t r;
    r.m00 = m00;
    r.m01 = m01;
    r.m02 = m02;
    r.m10 = m10;
    r.m11 = m11;
    r.m12 = m12;
    r.m20 = m20;
    r.m21 = m21;
    r.m22 = m22;
    return r;
}

static inline i3_mat3_t i3_mat3_set(float v)
{
    i3_mat3_t r;
    r.m00 = v;
    r.m01 = v;
    r.m02 = v;
    r.m10 = v;
    r.m11 = v;
    r.m12 = v;
    r.m20 = v;
    r.m21 = v;
    r.m22 = v;
    return r;
}

static inline i3_mat3_t i3_mat3_zero()
{
    return i3_mat3_set(0);
}

static inline i3_mat3_t i3_mat3_identity()
{
    i3_mat3_t r;
    r.m00 = 1;
    r.m01 = 0;
    r.m02 = 0;
    r.m10 = 0;
    r.m11 = 1;
    r.m12 = 0;
    r.m20 = 0;
    r.m21 = 0;
    r.m22 = 1;
    return r;
}

static inline i3_mat3_t i3_mat3_transpose(i3_mat3_t m)
{
    i3_mat3_t r;
    r.m00 = m.m00;
    r.m01 = m.m10;
    r.m02 = m.m20;
    r.m10 = m.m01;
    r.m11 = m.m11;
    r.m12 = m.m21;
    r.m20 = m.m02;
    r.m21 = m.m12;
    r.m22 = m.m22;
    return r;
}

static inline i3_mat3_t i3_mat3_mult(i3_mat3_t a, i3_mat3_t b)
{
    i3_mat3_t r;
    r.m00 = a.m00 * b.m00 + a.m01 * b.m10 + a.m02 * b.m20;
    r.m01 = a.m00 * b.m01 + a.m01 * b.m11 + a.m02 * b.m21;
    r.m02 = a.m00 * b.m02 + a.m01 * b.m12 + a.m02 * b.m22;
    r.m10 = a.m10 * b.m00 + a.m11 * b.m10 + a.m12 * b.m20;
    r.m11 = a.m10 * b.m01 + a.m11 * b.m11 + a.m12 * b.m21;
    r.m12 = a.m10 * b.m02 + a.m11 * b.m12 + a.m12 * b.m22;
    r.m20 = a.m20 * b.m00 + a.m21 * b.m10 + a.m22 * b.m20;
    r.m21 = a.m20 * b.m01 + a.m21 * b.m11 + a.m22 * b.m21;
    r.m22 = a.m20 * b.m02 + a.m21 * b.m12 + a.m22 * b.m22;
    return r;
}

static inline i3_mat34_t i3_mat3_mult_mat34(i3_mat3_t a, i3_mat34_t b)
{
    i3_mat34_t r;
    r.m00 = a.m00 * b.m00 + a.m01 * b.m10 + a.m02 * b.m20;
    r.m01 = a.m00 * b.m01 + a.m01 * b.m11 + a.m02 * b.m21;
    r.m02 = a.m00 * b.m02 + a.m01 * b.m12 + a.m02 * b.m22;
    r.m03 = a.m00 * b.m03 + a.m01 * b.m13 + a.m02 * b.m23;
    r.m10 = a.m10 * b.m00 + a.m11 * b.m10 + a.m12 * b.m20;
    r.m11 = a.m10 * b.m01 + a.m11 * b.m11 + a.m12 * b.m21;
    r.m12 = a.m10 * b.m02 + a.m11 * b.m12 + a.m12 * b.m22;
    r.m13 = a.m10 * b.m03 + a.m11 * b.m13 + a.m12 * b.m23;
    r.m20 = a.m20 * b.m00 + a.m21 * b.m10 + a.m22 * b.m20;
    r.m21 = a.m20 * b.m01 + a.m21 * b.m11 + a.m22 * b.m21;
    r.m22 = a.m20 * b.m02 + a.m21 * b.m12 + a.m22 * b.m22;
    r.m23 = a.m20 * b.m03 + a.m21 * b.m13 + a.m22 * b.m23;
    return r;
}

static inline i3_vec3_t i3_mat3_mult_vec3(i3_mat3_t m, i3_vec3_t v)
{
    i3_vec3_t r;
    r.x = m.m00 * v.x + m.m01 * v.y + m.m02 * v.z;
    r.y = m.m10 * v.x + m.m11 * v.y + m.m12 * v.z;
    r.z = m.m20 * v.x + m.m21 * v.y + m.m22 * v.z;
    return r;
}

static inline i3_vec3_t i3_vec3_mult_mat3(i3_vec3_t v, i3_mat3_t m)
{
    i3_vec3_t r;
    r.x = v.x * m.m00 + v.y * m.m10 + v.z * m.m20;
    r.y = v.x * m.m01 + v.y * m.m11 + v.z * m.m21;
    r.z = v.x * m.m02 + v.y * m.m12 + v.z * m.m22;
    return r;
}

static inline i3_mat2_t i3_mat3_submat(i3_mat3_t m, int row, int col)
{
    i3_mat2_t r;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        {
            int ii = i < row ? i : i + 1;
            int jj = j < col ? j : j + 1;
            r.m[i * 2 + j] = m.m[ii * 3 + jj];
        }
    return r;
}

static inline float i3_mat3_minor(i3_mat3_t m, int row, int col)
{
    return i3_mat2_det(i3_mat3_submat(m, row, col));
}

static inline float i3_mat3_det(i3_mat3_t m)
{
    return m.m00 * i3_mat3_minor(m, 0, 0) - m.m01 * i3_mat3_minor(m, 0, 1) + m.m02 * i3_mat3_minor(m, 0, 2);
}

static inline i3_mat3_t i3_mat3_invert(i3_mat3_t m)
{
    i3_mat3_t c, r;
    c.m00 = i3_mat3_minor(m, 0, 0);
    c.m01 = -i3_mat3_minor(m, 0, 1);
    c.m02 = i3_mat3_minor(m, 0, 2);
    c.m10 = -i3_mat3_minor(m, 1, 0);
    c.m11 = i3_mat3_minor(m, 1, 1);
    c.m12 = -i3_mat3_minor(m, 1, 2);
    c.m20 = i3_mat3_minor(m, 2, 0);
    c.m21 = -i3_mat3_minor(m, 2, 1);
    c.m22 = i3_mat3_minor(m, 2, 2);
    float det = m.m00 * c.m00 + m.m01 * c.m01 + m.m02 * c.m02;
    r.m00 = c.m00 / det;
    r.m01 = c.m10 / det;
    r.m02 = c.m20 / det;
    r.m10 = c.m01 / det;
    r.m11 = c.m11 / det;
    r.m12 = c.m21 / det;
    r.m20 = c.m02 / det;
    r.m21 = c.m12 / det;
    r.m22 = c.m22 / det;
    return r;
}

static inline bool i3_mat3_eq(i3_mat3_t a, i3_mat3_t b, float epsilon)
{
    for (int i = 0; i < 9; ++i)
        if (!i3_eqf(a.m[i], b.m[i], epsilon))
            return false;
    return true;
}

static inline bool i3_mat3_neq(i3_mat3_t a, i3_mat3_t b, float epsilon)
{
    return !i3_mat3_eq(a, b, epsilon);
}

static inline const char* i3_mat3_str(i3_mat3_t m)
{
    static char buffer[144];
    snprintf(buffer, sizeof(buffer), "{{%f, %f, %f}, {%f, %f, %f}, {%f, %f, %f}}", m.m00, m.m01, m.m02, m.m10, m.m11,
             m.m12, m.m20, m.m21, m.m22);
    return buffer;
}

// implementation of i3_mat34_t
static inline i3_mat34_t i3_mat34(float m00,
                                  float m01,
                                  float m02,
                                  float m03,
                                  float m10,
                                  float m11,
                                  float m12,
                                  float m13,
                                  float m20,
                                  float m21,
                                  float m22,
                                  float m23)
{
    i3_mat34_t r;
    r.m00 = m00;
    r.m01 = m01;
    r.m02 = m02;
    r.m03 = m03;
    r.m10 = m10;
    r.m11 = m11;
    r.m12 = m12;
    r.m13 = m13;
    r.m20 = m20;
    r.m21 = m21;
    r.m22 = m22;
    r.m23 = m23;
    return r;
}

static inline i3_mat34_t i3_mat34_set(float v)
{
    i3_mat34_t r;
    r.m00 = v;
    r.m01 = v;
    r.m02 = v;
    r.m03 = v;
    r.m10 = v;
    r.m11 = v;
    r.m12 = v;
    r.m13 = v;
    r.m20 = v;
    r.m21 = v;
    r.m22 = v;
    r.m23 = v;
    return r;
}

static inline i3_mat34_t i3_mat34_zero()
{
    return i3_mat34_set(0);
}

static inline i3_mat43_t i3_mat34_transpose(i3_mat34_t m)
{
    i3_mat43_t r;
    r.m00 = m.m00;
    r.m01 = m.m10;
    r.m02 = m.m20;
    r.m10 = m.m01;
    r.m11 = m.m11;
    r.m12 = m.m21;
    r.m20 = m.m02;
    r.m21 = m.m12;
    r.m22 = m.m22;
    r.m30 = m.m03;
    r.m31 = m.m13;
    r.m32 = m.m23;
    return r;
}

static inline i3_mat3_t i3_mat34_mult_mat43(i3_mat34_t a, i3_mat43_t b)
{
    i3_mat3_t r;
    r.m00 = a.m00 * b.m00 + a.m01 * b.m10 + a.m02 * b.m20 + a.m03 * b.m30;
    r.m01 = a.m00 * b.m01 + a.m01 * b.m11 + a.m02 * b.m21 + a.m03 * b.m31;
    r.m02 = a.m00 * b.m02 + a.m01 * b.m12 + a.m02 * b.m22 + a.m03 * b.m32;
    r.m10 = a.m10 * b.m00 + a.m11 * b.m10 + a.m12 * b.m20 + a.m13 * b.m30;
    r.m11 = a.m10 * b.m01 + a.m11 * b.m11 + a.m12 * b.m21 + a.m13 * b.m31;
    r.m12 = a.m10 * b.m02 + a.m11 * b.m12 + a.m12 * b.m22 + a.m13 * b.m32;
    r.m20 = a.m20 * b.m00 + a.m21 * b.m10 + a.m22 * b.m20 + a.m23 * b.m30;
    r.m21 = a.m20 * b.m01 + a.m21 * b.m11 + a.m22 * b.m21 + a.m23 * b.m31;
    r.m22 = a.m20 * b.m02 + a.m21 * b.m12 + a.m22 * b.m22 + a.m23 * b.m32;
    return r;
}

static inline i3_mat34_t i3_mat34_mult_mat4(i3_mat34_t a, i3_mat4_t b)
{
    i3_mat34_t r;
    r.m00 = a.m00 * b.m00 + a.m01 * b.m10 + a.m02 * b.m20 + a.m03 * b.m30;
    r.m01 = a.m00 * b.m01 + a.m01 * b.m11 + a.m02 * b.m21 + a.m03 * b.m31;
    r.m02 = a.m00 * b.m02 + a.m01 * b.m12 + a.m02 * b.m22 + a.m03 * b.m32;
    r.m03 = a.m00 * b.m03 + a.m01 * b.m13 + a.m02 * b.m23 + a.m03 * b.m33;
    r.m10 = a.m10 * b.m00 + a.m11 * b.m10 + a.m12 * b.m20 + a.m13 * b.m30;
    r.m11 = a.m10 * b.m01 + a.m11 * b.m11 + a.m12 * b.m21 + a.m13 * b.m31;
    r.m12 = a.m10 * b.m02 + a.m11 * b.m12 + a.m12 * b.m22 + a.m13 * b.m32;
    r.m13 = a.m10 * b.m03 + a.m11 * b.m13 + a.m12 * b.m23 + a.m13 * b.m33;
    r.m20 = a.m20 * b.m00 + a.m21 * b.m10 + a.m22 * b.m20 + a.m23 * b.m30;
    r.m21 = a.m20 * b.m01 + a.m21 * b.m11 + a.m22 * b.m21 + a.m23 * b.m31;
    r.m22 = a.m20 * b.m02 + a.m21 * b.m12 + a.m22 * b.m22 + a.m23 * b.m32;
    r.m23 = a.m20 * b.m03 + a.m21 * b.m13 + a.m22 * b.m23 + a.m23 * b.m33;
    return r;
}

static inline i3_vec3_t i3_mat34_mult_vec4(i3_mat34_t m, i3_vec4_t v)
{
    i3_vec3_t r;
    r.x = m.m00 * v.x + m.m01 * v.y + m.m02 * v.z + m.m03 * v.w;
    r.y = m.m10 * v.x + m.m11 * v.y + m.m12 * v.z + m.m13 * v.w;
    r.z = m.m20 * v.x + m.m21 * v.y + m.m22 * v.z + m.m23 * v.w;
    return r;
}

static inline i3_vec4_t i3_vec3_mult_mat34(i3_vec3_t v, i3_mat34_t m)
{
    i3_vec4_t r;
    r.x = v.x * m.m00 + v.y * m.m10 + v.z * m.m20;
    r.y = v.x * m.m01 + v.y * m.m11 + v.z * m.m21;
    r.z = v.x * m.m02 + v.y * m.m12 + v.z * m.m22;
    r.w = v.x * m.m03 + v.y * m.m13 + v.z * m.m23;
    return r;
}

static inline bool i3_mat34_eq(i3_mat34_t a, i3_mat34_t b, float epsilon)
{
    for (int i = 0; i < 12; ++i)
        if (!i3_eqf(a.m[i], b.m[i], epsilon))
            return false;
    return true;
}

static inline bool i3_mat34_neq(i3_mat34_t a, i3_mat34_t b, float epsilon)
{
    return !i3_mat34_eq(a, b, epsilon);
}

static inline const char* i3_mat34_str(i3_mat34_t m)
{
    static char buffer[192];
    snprintf(buffer, sizeof(buffer), "{{%f, %f, %f, %f}, {%f, %f, %f, %f}, {%f, %f, %f, %f}}", m.m00, m.m01, m.m02,
             m.m03, m.m10, m.m11, m.m12, m.m13, m.m20, m.m21, m.m22, m.m23);
    return buffer;
}

// implementation of i3_mat43_t
static inline i3_mat43_t i3_mat43(float m00,
                                  float m01,
                                  float m02,
                                  float m10,
                                  float m11,
                                  float m12,
                                  float m20,
                                  float m21,
                                  float m22,
                                  float m30,
                                  float m31,
                                  float m32)
{
    i3_mat43_t r;
    r.m00 = m00;
    r.m01 = m01;
    r.m02 = m02;
    r.m10 = m10;
    r.m11 = m11;
    r.m12 = m12;
    r.m20 = m20;
    r.m21 = m21;
    r.m22 = m22;
    r.m30 = m30;
    r.m31 = m31;
    r.m32 = m32;
    return r;
}

static inline i3_mat43_t i3_mat43_set(float v)
{
    i3_mat43_t r;
    r.m00 = v;
    r.m01 = v;
    r.m02 = v;
    r.m10 = v;
    r.m11 = v;
    r.m12 = v;
    r.m20 = v;
    r.m21 = v;
    r.m22 = v;
    r.m30 = v;
    r.m31 = v;
    r.m32 = v;
    return r;
}

static inline i3_mat43_t i3_mat43_zero()
{
    return i3_mat43_set(0);
}

static inline i3_mat34_t i3_mat43_transpose(i3_mat43_t m)
{
    i3_mat34_t r;
    r.m00 = m.m00;
    r.m01 = m.m10;
    r.m02 = m.m20;
    r.m03 = m.m30;
    r.m10 = m.m01;
    r.m11 = m.m11;
    r.m12 = m.m21;
    r.m13 = m.m31;
    r.m20 = m.m02;
    r.m21 = m.m12;
    r.m22 = m.m22;
    r.m23 = m.m32;
    return r;
}

static inline i3_mat43_t i3_mat43_mult_mat3(i3_mat43_t a, i3_mat3_t b)
{
    i3_mat43_t r;
    r.m00 = a.m00 * b.m00 + a.m01 * b.m10 + a.m02 * b.m20;
    r.m01 = a.m00 * b.m01 + a.m01 * b.m11 + a.m02 * b.m21;
    r.m02 = a.m00 * b.m02 + a.m01 * b.m12 + a.m02 * b.m22;
    r.m10 = a.m10 * b.m00 + a.m11 * b.m10 + a.m12 * b.m20;
    r.m11 = a.m10 * b.m01 + a.m11 * b.m11 + a.m12 * b.m21;
    r.m12 = a.m10 * b.m02 + a.m11 * b.m12 + a.m12 * b.m22;
    r.m20 = a.m20 * b.m00 + a.m21 * b.m10 + a.m22 * b.m20;
    r.m21 = a.m20 * b.m01 + a.m21 * b.m11 + a.m22 * b.m21;
    r.m22 = a.m20 * b.m02 + a.m21 * b.m12 + a.m22 * b.m22;
    r.m30 = a.m30 * b.m00 + a.m31 * b.m10 + a.m32 * b.m20;
    r.m31 = a.m30 * b.m01 + a.m31 * b.m11 + a.m32 * b.m21;
    r.m32 = a.m30 * b.m02 + a.m31 * b.m12 + a.m32 * b.m22;
    return r;
}

static inline i3_mat4_t i3_mat43_mult_mat34(i3_mat43_t a, i3_mat34_t b)
{
    i3_mat4_t r;
    r.m00 = a.m00 * b.m00 + a.m01 * b.m10 + a.m02 * b.m20;
    r.m01 = a.m00 * b.m01 + a.m01 * b.m11 + a.m02 * b.m21;
    r.m02 = a.m00 * b.m02 + a.m01 * b.m12 + a.m02 * b.m22;
    r.m03 = a.m00 * b.m03 + a.m01 * b.m13 + a.m02 * b.m23;
    r.m10 = a.m10 * b.m00 + a.m11 * b.m10 + a.m12 * b.m20;
    r.m11 = a.m10 * b.m01 + a.m11 * b.m11 + a.m12 * b.m21;
    r.m12 = a.m10 * b.m02 + a.m11 * b.m12 + a.m12 * b.m22;
    r.m13 = a.m10 * b.m03 + a.m11 * b.m13 + a.m12 * b.m23;
    r.m20 = a.m20 * b.m00 + a.m21 * b.m10 + a.m22 * b.m20;
    r.m21 = a.m20 * b.m01 + a.m21 * b.m11 + a.m22 * b.m21;
    r.m22 = a.m20 * b.m02 + a.m21 * b.m12 + a.m22 * b.m22;
    r.m23 = a.m20 * b.m03 + a.m21 * b.m13 + a.m22 * b.m23;
    r.m30 = a.m30 * b.m00 + a.m31 * b.m10 + a.m32 * b.m20;
    r.m31 = a.m30 * b.m01 + a.m31 * b.m11 + a.m32 * b.m21;
    r.m32 = a.m30 * b.m02 + a.m31 * b.m12 + a.m32 * b.m22;
    r.m33 = a.m30 * b.m03 + a.m31 * b.m13 + a.m32 * b.m23;
    return r;
}

static inline i3_vec4_t i3_mat43_mult_vec3(i3_mat43_t m, i3_vec3_t v)
{
    i3_vec4_t r;
    r.x = m.m00 * v.x + m.m01 * v.y + m.m02 * v.z;
    r.y = m.m10 * v.x + m.m11 * v.y + m.m12 * v.z;
    r.z = m.m20 * v.x + m.m21 * v.y + m.m22 * v.z;
    r.w = m.m30 * v.x + m.m31 * v.y + m.m32 * v.z;
    return r;
}

static inline i3_vec3_t i3_vec4_mult_mat43(i3_vec4_t v, i3_mat43_t m)
{
    i3_vec3_t r;
    r.x = v.x * m.m00 + v.y * m.m10 + v.z * m.m20 + v.w * m.m30;
    r.y = v.x * m.m01 + v.y * m.m11 + v.z * m.m21 + v.w * m.m31;
    r.z = v.x * m.m02 + v.y * m.m12 + v.z * m.m22 + v.w * m.m32;
    return r;
}

static inline bool i3_mat43_eq(i3_mat43_t a, i3_mat43_t b, float epsilon)
{
    for (int i = 0; i < 12; ++i)
        if (!i3_eqf(a.m[i], b.m[i], epsilon))
            return false;
    return true;
}

static inline bool i3_mat43_neq(i3_mat43_t a, i3_mat43_t b, float epsilon)
{
    return !i3_mat43_eq(a, b, epsilon);
}

static inline const char* i3_mat43_str(i3_mat43_t m)
{
    static char buffer[192];
    snprintf(buffer, sizeof(buffer), "{{%f, %f, %f}, {%f, %f, %f}, {%f, %f, %f}, {%f, %f, %f}}", m.m00, m.m01, m.m02,
             m.m10, m.m11, m.m12, m.m20, m.m21, m.m22, m.m30, m.m31, m.m32);
    return buffer;
}

// implementation of i3_mat4_t
static inline i3_mat4_t i3_mat4(float m00,
                                float m01,
                                float m02,
                                float m03,
                                float m10,
                                float m11,
                                float m12,
                                float m13,
                                float m20,
                                float m21,
                                float m22,
                                float m23,
                                float m30,
                                float m31,
                                float m32,
                                float m33)
{
    i3_mat4_t r;
    r.m00 = m00;
    r.m01 = m01;
    r.m02 = m02;
    r.m03 = m03;
    r.m10 = m10;
    r.m11 = m11;
    r.m12 = m12;
    r.m13 = m13;
    r.m20 = m20;
    r.m21 = m21;
    r.m22 = m22;
    r.m23 = m23;
    r.m30 = m30;
    r.m31 = m31;
    r.m32 = m32;
    r.m33 = m33;
    return r;
}

static inline i3_mat4_t i3_mat4_set(float v)
{
    i3_mat4_t r;
    r.m00 = v;
    r.m01 = v;
    r.m02 = v;
    r.m03 = v;
    r.m10 = v;
    r.m11 = v;
    r.m12 = v;
    r.m13 = v;
    r.m20 = v;
    r.m21 = v;
    r.m22 = v;
    r.m23 = v;
    r.m30 = v;
    r.m31 = v;
    r.m32 = v;
    r.m33 = v;
    return r;
}

static inline i3_mat4_t i3_mat4_zero()
{
    return i3_mat4_set(0);
}

static inline i3_mat4_t i3_mat4_identity()
{
    i3_mat4_t r;
    r.m00 = 1;
    r.m01 = 0;
    r.m02 = 0;
    r.m03 = 0;
    r.m10 = 0;
    r.m11 = 1;
    r.m12 = 0;
    r.m13 = 0;
    r.m20 = 0;
    r.m21 = 0;
    r.m22 = 1;
    r.m23 = 0;
    r.m30 = 0;
    r.m31 = 0;
    r.m32 = 0;
    r.m33 = 1;
    return r;
}

static inline i3_mat4_t i3_mat4_transpose(i3_mat4_t m)
{
    i3_mat4_t r;
    r.m00 = m.m00;
    r.m01 = m.m10;
    r.m02 = m.m20;
    r.m03 = m.m30;
    r.m10 = m.m01;
    r.m11 = m.m11;
    r.m12 = m.m21;
    r.m13 = m.m31;
    r.m20 = m.m02;
    r.m21 = m.m12;
    r.m22 = m.m22;
    r.m23 = m.m32;
    r.m30 = m.m03;
    r.m31 = m.m13;
    r.m32 = m.m23;
    r.m33 = m.m33;
    return r;
}

static inline i3_mat43_t i3_mat4_mult_mat43(i3_mat4_t a, i3_mat43_t b)
{
    i3_mat43_t r;
    r.m00 = a.m00 * b.m00 + a.m01 * b.m10 + a.m02 * b.m20 + a.m03 * b.m30;
    r.m01 = a.m00 * b.m01 + a.m01 * b.m11 + a.m02 * b.m21 + a.m03 * b.m31;
    r.m02 = a.m00 * b.m02 + a.m01 * b.m12 + a.m02 * b.m22 + a.m03 * b.m32;
    r.m10 = a.m10 * b.m00 + a.m11 * b.m10 + a.m12 * b.m20 + a.m13 * b.m30;
    r.m11 = a.m10 * b.m01 + a.m11 * b.m11 + a.m12 * b.m21 + a.m13 * b.m31;
    r.m12 = a.m10 * b.m02 + a.m11 * b.m12 + a.m12 * b.m22 + a.m13 * b.m32;
    r.m20 = a.m20 * b.m00 + a.m21 * b.m10 + a.m22 * b.m20 + a.m23 * b.m30;
    r.m21 = a.m20 * b.m01 + a.m21 * b.m11 + a.m22 * b.m21 + a.m23 * b.m31;
    r.m22 = a.m20 * b.m02 + a.m21 * b.m12 + a.m22 * b.m22 + a.m23 * b.m32;
    r.m30 = a.m30 * b.m00 + a.m31 * b.m10 + a.m32 * b.m20 + a.m33 * b.m30;
    r.m31 = a.m30 * b.m01 + a.m31 * b.m11 + a.m32 * b.m21 + a.m33 * b.m31;
    r.m32 = a.m30 * b.m02 + a.m31 * b.m12 + a.m32 * b.m22 + a.m33 * b.m32;
    return r;
}

static inline i3_mat4_t i3_mat4_mult(i3_mat4_t a, i3_mat4_t b)
{
    i3_mat4_t r;
    r.m00 = a.m00 * b.m00 + a.m01 * b.m10 + a.m02 * b.m20 + a.m03 * b.m30;
    r.m01 = a.m00 * b.m01 + a.m01 * b.m11 + a.m02 * b.m21 + a.m03 * b.m31;
    r.m02 = a.m00 * b.m02 + a.m01 * b.m12 + a.m02 * b.m22 + a.m03 * b.m32;
    r.m03 = a.m00 * b.m03 + a.m01 * b.m13 + a.m02 * b.m23 + a.m03 * b.m33;
    r.m10 = a.m10 * b.m00 + a.m11 * b.m10 + a.m12 * b.m20 + a.m13 * b.m30;
    r.m11 = a.m10 * b.m01 + a.m11 * b.m11 + a.m12 * b.m21 + a.m13 * b.m31;
    r.m12 = a.m10 * b.m02 + a.m11 * b.m12 + a.m12 * b.m22 + a.m13 * b.m32;
    r.m13 = a.m10 * b.m03 + a.m11 * b.m13 + a.m12 * b.m23 + a.m13 * b.m33;
    r.m20 = a.m20 * b.m00 + a.m21 * b.m10 + a.m22 * b.m20 + a.m23 * b.m30;
    r.m21 = a.m20 * b.m01 + a.m21 * b.m11 + a.m22 * b.m21 + a.m23 * b.m31;
    r.m22 = a.m20 * b.m02 + a.m21 * b.m12 + a.m22 * b.m22 + a.m23 * b.m32;
    r.m23 = a.m20 * b.m03 + a.m21 * b.m13 + a.m22 * b.m23 + a.m23 * b.m33;
    r.m30 = a.m30 * b.m00 + a.m31 * b.m10 + a.m32 * b.m20 + a.m33 * b.m30;
    r.m31 = a.m30 * b.m01 + a.m31 * b.m11 + a.m32 * b.m21 + a.m33 * b.m31;
    r.m32 = a.m30 * b.m02 + a.m31 * b.m12 + a.m32 * b.m22 + a.m33 * b.m32;
    r.m33 = a.m30 * b.m03 + a.m31 * b.m13 + a.m32 * b.m23 + a.m33 * b.m33;
    return r;
}

static inline i3_vec4_t i3_mat4_mult_vec4(i3_mat4_t m, i3_vec4_t v)
{
    i3_vec4_t r;
    r.x = m.m00 * v.x + m.m01 * v.y + m.m02 * v.z + m.m03 * v.w;
    r.y = m.m10 * v.x + m.m11 * v.y + m.m12 * v.z + m.m13 * v.w;
    r.z = m.m20 * v.x + m.m21 * v.y + m.m22 * v.z + m.m23 * v.w;
    r.w = m.m30 * v.x + m.m31 * v.y + m.m32 * v.z + m.m33 * v.w;
    return r;
}

static inline i3_vec4_t i3_vec4_mult_mat4(i3_vec4_t v, i3_mat4_t m)
{
    i3_vec4_t r;
    r.x = v.x * m.m00 + v.y * m.m10 + v.z * m.m20 + v.w * m.m30;
    r.y = v.x * m.m01 + v.y * m.m11 + v.z * m.m21 + v.w * m.m31;
    r.z = v.x * m.m02 + v.y * m.m12 + v.z * m.m22 + v.w * m.m32;
    r.w = v.x * m.m03 + v.y * m.m13 + v.z * m.m23 + v.w * m.m33;
    return r;
}

static inline i3_mat3_t i3_mat4_submat(i3_mat4_t m, int row, int col)
{
    i3_mat3_t r;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            int ii = i < row ? i : i + 1;
            int jj = j < col ? j : j + 1;
            r.m[i * 3 + j] = m.m[ii * 4 + jj];
        }
    return r;
}

static inline float i3_mat4_minor(i3_mat4_t m, int row, int col)
{
    return i3_mat3_det(i3_mat4_submat(m, row, col));
}

static inline float i3_mat4_det(i3_mat4_t m)
{
    return m.m00 * i3_mat4_minor(m, 0, 0) - m.m01 * i3_mat4_minor(m, 0, 1) + m.m02 * i3_mat4_minor(m, 0, 2) -
           m.m03 * i3_mat4_minor(m, 0, 3);
}

static inline i3_mat4_t i3_mat4_invert(i3_mat4_t m)
{
    i3_mat4_t c, r;
    c.m00 = i3_mat4_minor(m, 0, 0);
    c.m01 = -i3_mat4_minor(m, 0, 1);
    c.m02 = i3_mat4_minor(m, 0, 2);
    c.m03 = -i3_mat4_minor(m, 0, 3);
    c.m10 = -i3_mat4_minor(m, 1, 0);
    c.m11 = i3_mat4_minor(m, 1, 1);
    c.m12 = -i3_mat4_minor(m, 1, 2);
    c.m13 = i3_mat4_minor(m, 1, 3);
    c.m20 = i3_mat4_minor(m, 2, 0);
    c.m21 = -i3_mat4_minor(m, 2, 1);
    c.m22 = i3_mat4_minor(m, 2, 2);
    c.m23 = -i3_mat4_minor(m, 2, 3);
    c.m30 = -i3_mat4_minor(m, 3, 0);
    c.m31 = i3_mat4_minor(m, 3, 1);
    c.m32 = -i3_mat4_minor(m, 3, 2);
    c.m33 = i3_mat4_minor(m, 3, 3);
    float det = m.m00 * c.m00 + m.m01 * c.m01 + m.m02 * c.m02 + m.m03 * c.m03;
    r.m00 = c.m00 / det;
    r.m01 = c.m10 / det;
    r.m02 = c.m20 / det;
    r.m03 = c.m30 / det;
    r.m10 = c.m01 / det;
    r.m11 = c.m11 / det;
    r.m12 = c.m21 / det;
    r.m13 = c.m31 / det;
    r.m20 = c.m02 / det;
    r.m21 = c.m12 / det;
    r.m22 = c.m22 / det;
    r.m23 = c.m32 / det;
    r.m30 = c.m03 / det;
    r.m31 = c.m13 / det;
    r.m32 = c.m23 / det;
    r.m33 = c.m33 / det;
    return r;
}

static inline bool i3_mat4_eq(i3_mat4_t a, i3_mat4_t b, float epsilon)
{
    for (int i = 0; i < 16; ++i)
        if (!i3_eqf(a.m[i], b.m[i], epsilon))
            return false;
    return true;
}

static inline bool i3_mat4_neq(i3_mat4_t a, i3_mat4_t b, float epsilon)
{
    return !i3_mat4_eq(a, b, epsilon);
}

static inline const char* i3_mat4_str(i3_mat4_t m)
{
    static char buffer[256];
    snprintf(buffer, sizeof(buffer), "{{%f, %f, %f, %f}, {%f, %f, %f, %f}, {%f, %f, %f, %f}, {%f, %f, %f, %f}}", m.m00,
             m.m01, m.m02, m.m03, m.m10, m.m11, m.m12, m.m13, m.m20, m.m21, m.m22, m.m23, m.m30, m.m31, m.m32, m.m33);
    return buffer;
}
