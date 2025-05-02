#pragma once

#include "common.h"

// vec2
typedef struct i3_vec2_t
{
    union
    {
        float v[2];
        struct
        {
            float x, y;
        };
    };
} i3_vec2_t;

static inline i3_vec2_t i3_vec2(float x, float y)
{
    i3_vec2_t vec;
    vec.x = x;
    vec.y = y;
    return vec;
}

static inline const char* i3_vec2_dump(i3_vec2_t vec)
{
    static char buf[64];
    snprintf(buf, sizeof(buf), "[%f %f]", vec.x, vec.y);
    return buf;
}

// vec3
typedef struct i3_vec3_t
{
    union
    {
        float v[3];
        struct
        {
            float x, y, z;
        };
    };
} i3_vec3_t;

static inline i3_vec3_t i3_vec3(float x, float y, float z)
{
    i3_vec3_t vec;
    vec.x = x;
    vec.y = y;
    vec.z = z;
    return vec;
}

static inline const char* i3_vec3_dump(i3_vec3_t vec)
{
    static char buf[64];
    snprintf(buf, sizeof(buf), "[%f %f %f]", vec.x, vec.y, vec.z);
    return buf;
}

// vec4
typedef struct i3_vec4_t
{
    union
    {
        float v[4];
        struct
        {
            float x, y, z, w;
        };
    };
} i3_vec4_t;

static inline i3_vec4_t i3_vec4(float x, float y, float z, float w)
{
    i3_vec4_t vec;
    vec.x = x;
    vec.y = y;
    vec.z = z;
    vec.w = w;
    return vec;
}

static inline const char* i3_vec4_dump(i3_vec4_t vec)
{
    static char buf[64];
    snprintf(buf, sizeof(buf), "[%f %f %f %f]", vec.x, vec.y, vec.z, vec.w);
    return buf;
}

#define I3_VEC(size, suffix) i3_vec##size##_##suffix
#define I3_VEC_T(size) I3_VEC(size, t)
#define I3_VEC_FOR(size) for (int i = 0; i < size; i++)

#define I3_DECL_VEC(size)                                                                    \
    static inline I3_VEC_T(size) I3_VEC(size, set)(float v)                                  \
    {                                                                                        \
        I3_VEC_T(size) r;                                                                    \
        I3_VEC_FOR(size) r.v[i] = v;                                                         \
        return r;                                                                            \
    }                                                                                        \
                                                                                             \
    static inline I3_VEC_T(size) I3_VEC(size, zero)(void)                                    \
    {                                                                                        \
        return I3_VEC(size, set)(0.0f);                                                      \
    }                                                                                        \
                                                                                             \
    static inline I3_VEC_T(size) I3_VEC(size, one)(void)                                     \
    {                                                                                        \
        return I3_VEC(size, set)(1.0f);                                                      \
    }                                                                                        \
                                                                                             \
    static inline I3_VEC_T(size) I3_VEC(size, abs)(I3_VEC_T(size) a)                         \
    {                                                                                        \
        I3_VEC_T(size) r;                                                                    \
        I3_VEC_FOR(size) r.v[i] = i3_absf(a.v[i]);                                           \
        return r;                                                                            \
    }                                                                                        \
                                                                                             \
    static inline I3_VEC_T(size) I3_VEC(size, neg)(I3_VEC_T(size) a)                         \
    {                                                                                        \
        I3_VEC_T(size) r;                                                                    \
        I3_VEC_FOR(size) r.v[i] = -a.v[i];                                                   \
        return r;                                                                            \
    }                                                                                        \
                                                                                             \
    static inline I3_VEC_T(size) I3_VEC(size, add)(I3_VEC_T(size) a, I3_VEC_T(size) b)       \
    {                                                                                        \
        I3_VEC_T(size) r;                                                                    \
        I3_VEC_FOR(size) r.v[i] = a.v[i] + b.v[i];                                           \
        return r;                                                                            \
    }                                                                                        \
                                                                                             \
    static inline I3_VEC_T(size) I3_VEC(size, sub)(I3_VEC_T(size) a, I3_VEC_T(size) b)       \
    {                                                                                        \
        I3_VEC_T(size) r;                                                                    \
        I3_VEC_FOR(size) r.v[i] = a.v[i] - b.v[i];                                           \
        return r;                                                                            \
    }                                                                                        \
                                                                                             \
    static inline I3_VEC_T(size) I3_VEC(size, mul)(I3_VEC_T(size) a, I3_VEC_T(size) b)       \
    {                                                                                        \
        I3_VEC_T(size) r;                                                                    \
        I3_VEC_FOR(size) r.v[i] = a.v[i] * b.v[i];                                           \
        return r;                                                                            \
    }                                                                                        \
                                                                                             \
    static inline I3_VEC_T(size) I3_VEC(size, div)(I3_VEC_T(size) a, I3_VEC_T(size) b)       \
    {                                                                                        \
        I3_VEC_T(size) r;                                                                    \
        I3_VEC_FOR(size) r.v[i] = a.v[i] / b.v[i];                                           \
        return r;                                                                            \
    }                                                                                        \
                                                                                             \
    static inline I3_VEC_T(size) I3_VEC(size, scale)(I3_VEC_T(size) vec, float scale)        \
    {                                                                                        \
        I3_VEC_T(size) r;                                                                    \
        I3_VEC_FOR(size) r.v[i] = vec.v[i] * scale;                                          \
        return r;                                                                            \
    }                                                                                        \
    static inline float I3_VEC(size, dot)(I3_VEC_T(size) a, I3_VEC_T(size) b)                \
    {                                                                                        \
        float r = 0.0f;                                                                      \
        I3_VEC_FOR(size) r += a.v[i] * b.v[i];                                               \
        return r;                                                                            \
    }                                                                                        \
    static inline float I3_VEC(size, len2)(I3_VEC_T(size) a)                                 \
    {                                                                                        \
        return I3_VEC(size, dot)(a, a);                                                      \
    }                                                                                        \
    static inline float I3_VEC(size, len)(I3_VEC_T(size) a)                                  \
    {                                                                                        \
        return i3_sqrtf(I3_VEC(size, len2)(a));                                              \
    }                                                                                        \
                                                                                             \
    static inline I3_VEC_T(size) I3_VEC(size, normalize)(I3_VEC_T(size) a)                   \
    {                                                                                        \
        float len = I3_VEC(size, len)(a);                                                    \
        I3_VEC_T(size) r;                                                                    \
        I3_VEC_FOR(size) r.v[i] = a.v[i] / len;                                              \
        return r;                                                                            \
    }                                                                                        \
                                                                                             \
    static inline I3_VEC_T(size) I3_VEC(size, min)(I3_VEC_T(size) a, I3_VEC_T(size) b)       \
    {                                                                                        \
        I3_VEC_T(size) r;                                                                    \
        I3_VEC_FOR(size) r.v[i] = i3_minf(a.v[i], b.v[i]);                                   \
        return r;                                                                            \
    }                                                                                        \
    static inline I3_VEC_T(size) I3_VEC(size, max)(I3_VEC_T(size) a, I3_VEC_T(size) b)       \
    {                                                                                        \
        I3_VEC_T(size) r;                                                                    \
        I3_VEC_FOR(size) r.v[i] = i3_maxf(a.v[i], b.v[i]);                                   \
        return r;                                                                            \
    }                                                                                        \
                                                                                             \
    static inline I3_VEC_T(size) I3_VEC(size, clamp)(I3_VEC_T(size) a, float min, float max) \
    {                                                                                        \
        I3_VEC_T(size) r;                                                                    \
        I3_VEC_FOR(size) r.v[i] = i3_clampf(a.v[i], min, max);                               \
        return r;                                                                            \
    }                                                                                        \
                                                                                             \
    static inline I3_VEC_T(size) I3_VEC(size, saturate)(I3_VEC_T(size) a)                    \
    {                                                                                        \
        I3_VEC_T(size) r;                                                                    \
        I3_VEC_FOR(size) r.v[i] = i3_saturatef(a.v[i]);                                      \
        return r;                                                                            \
    }                                                                                        \
                                                                                             \
    static inline bool I3_VEC(size, eq)(I3_VEC_T(size) a, I3_VEC_T(size) b, float epsilon)   \
    {                                                                                        \
        I3_VEC_FOR(size)                                                                     \
        if (!i3_eqf(a.v[i], b.v[i], epsilon))                                                \
            return false;                                                                    \
        return true;                                                                         \
    }                                                                                        \
                                                                                             \
    static inline bool I3_VEC(size, neq)(I3_VEC_T(size) a, I3_VEC_T(size) b, float epsilon)  \
    {                                                                                        \
        return !I3_VEC(size, eq)(a, b, epsilon);                                             \
    }

I3_DECL_VEC(2)
I3_DECL_VEC(3)
I3_DECL_VEC(4)
