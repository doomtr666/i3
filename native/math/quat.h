#pragma once

#include "vec.h"

typedef struct i3_quat_t
{
    float a, b, c, d;
} i3_quat_t;

static inline i3_quat_t i3_quat(float a, float b, float c, float d);
static inline i3_quat_t i3_quat_identity();
static inline i3_quat_t i3_quat_axis_angle(i3_vec3_t axis, float angle);
static inline i3_quat_t i3_quat_scale(i3_quat_t q, float v);
static inline i3_quat_t i3_quat_mult(i3_quat_t q1, i3_quat_t q2);
static inline i3_vec3_t i3_quat_transform(i3_quat_t q, i3_vec3_t v);
static inline i3_quat_t i3_quat_conjugate(i3_quat_t q);
static inline i3_quat_t i3_quat_invert(i3_quat_t q);
static inline float i3_quat_len2(i3_quat_t q);
static inline float i3_quat_len(i3_quat_t q);
static inline i3_quat_t i3_quat_normalize(i3_quat_t q);
static inline bool i3_quat_eq(i3_quat_t q1, i3_quat_t q2, float epsilon);
static inline bool i3_quat_neq(i3_quat_t q1, i3_quat_t q2, float epsilon);
static inline const char* i3_quat_str(i3_quat_t q);

// implementation

static inline i3_quat_t i3_quat(float a, float b, float c, float d)
{
    i3_quat_t r;
    r.a = a;
    r.b = b;
    r.c = c;
    r.d = d;
    return r;
}

static inline i3_quat_t i3_quat_identity()
{
    i3_quat_t r;
    r.a = 1.0f;
    r.b = 0.0f;
    r.c = 0.0f;
    r.d = 0.0f;
    return r;
}

static inline i3_quat_t i3_quat_axis_angle(i3_vec3_t axis, float angle)
{
    i3_quat_t r;
    float half_angle = angle * 0.5f;
    r.a = i3_cosf(half_angle);
    float s = i3_sinf(half_angle);
    r.b = axis.x * s;
    r.c = axis.y * s;
    r.d = axis.z * s;
    return r;
}

static inline i3_quat_t i3_quat_scale(i3_quat_t q, float v)
{
    i3_quat_t r;
    r.a = q.a * v;
    r.b = q.b * v;
    r.c = q.c * v;
    r.d = q.d * v;
    return r;
}

static inline i3_quat_t i3_quat_mult(i3_quat_t q1, i3_quat_t q2)
{
    i3_quat_t r;
    r.a = q1.a * q2.a - q1.b * q2.b - q1.c * q2.c - q1.d * q2.d;
    r.b = q1.a * q2.b + q1.b * q2.a + q1.c * q2.d - q1.d * q2.c;
    r.c = q1.a * q2.c - q1.b * q2.d + q1.c * q2.a + q1.d * q2.b;
    r.d = q1.a * q2.d + q1.b * q2.c - q1.c * q2.b + q1.d * q2.a;
    return r;
}

static inline i3_vec3_t i3_quat_transform(i3_quat_t q, i3_vec3_t v)
{
    i3_vec3_t r;
    i3_quat_t qv = {0.0f, v.x, v.y, v.z};
    i3_quat_t qr = i3_quat_mult(i3_quat_mult(q, qv), i3_quat_invert(q));
    r.x = qr.b;
    r.y = qr.c;
    r.z = qr.d;
    return r;
}

static inline i3_quat_t i3_quat_conjugate(i3_quat_t q)
{
    i3_quat_t r;
    r.a = q.a;
    r.b = -q.b;
    r.c = -q.c;
    r.d = -q.d;
    return r;
}

static inline i3_quat_t i3_quat_invert(i3_quat_t q)
{
    float len = i3_quat_len(q);
    return i3_quat_scale(i3_quat_conjugate(q), 1.0f / len);
}

static inline float i3_quat_len2(i3_quat_t q)
{
    return q.a * q.a + q.b * q.b + q.c * q.c + q.d * q.d;
}

static inline float i3_quat_len(i3_quat_t q)
{
    return i3_sqrtf(i3_quat_len2(q));
}

static inline i3_quat_t i3_quat_normalize(i3_quat_t q)
{
    float len = i3_quat_len(q);
    i3_quat_t r;
    r.a = q.a / len;
    r.b = q.b / len;
    r.c = q.c / len;
    r.d = q.d / len;
    return r;
}

static inline bool i3_quat_eq(i3_quat_t q1, i3_quat_t q2, float epsilon)
{
    return i3_eqf(q1.a, q2.a, epsilon) && i3_eqf(q1.b, q2.b, epsilon) && i3_eqf(q1.c, q2.c, epsilon) &&
           i3_eqf(q1.d, q2.d, epsilon);
}

static inline bool i3_quat_neq(i3_quat_t q1, i3_quat_t q2, float epsilon)
{
    return !i3_quat_eq(q1, q2, epsilon);
}

static inline const char* i3_quat_str(i3_quat_t q)
{
    static char buffer[64];
    snprintf(buffer, sizeof(buffer), "(%f, %f, %f, %f)", q.a, q.b, q.c, q.d);
    return buffer;
}