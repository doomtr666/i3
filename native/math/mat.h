#pragma once

#include "vec.h"

#define I3_MAT(r, c, suffix) i3_mat##r##c##_##suffix
#define I3_MAT_T(r, c) I3_MAT(r, c, t)

#define I3_MAT_STRUCT(r, c)       \
    typedef struct I3_MAT_T(r, c) \
    {                             \
        float m[r * c];           \
    } I3_MAT_T(r, c);

#define I3_MAT_DECL(r, c)                                                                   \
    static inline I3_MAT_T(r, c) I3_MAT(r, c, set)(float v)                                 \
    {                                                                                       \
        I3_MAT_T(r, c) res;                                                                 \
        for (int i = 0; i < r * c; i++)                                                     \
            res.m[i] = v;                                                                   \
        return res;                                                                         \
    }                                                                                       \
                                                                                            \
    static inline I3_MAT_T(c, r) I3_MAT(r, c, transpose)(I3_MAT_T(r, c) a)                  \
    {                                                                                       \
        I3_MAT_T(c, r) res;                                                                 \
        for (int i = 0; i < r; i++)                                                         \
            for (int j = 0; j < c; j++)                                                     \
                res.m[j * r + i] = a.m[i * c + j];                                          \
        return res;                                                                         \
    }                                                                                       \
                                                                                            \
    static inline bool I3_MAT(r, c, eq)(I3_MAT_T(r, c) a, I3_MAT_T(r, c) b, float epsilon)  \
    {                                                                                       \
        for (int i = 0; i < r * c; i++)                                                     \
            if (!i3_eqf(a.m[i], b.m[i], epsilon))                                           \
                return false;                                                               \
        return true;                                                                        \
    }                                                                                       \
                                                                                            \
    static inline bool I3_MAT(r, c, neq)(I3_MAT_T(r, c) a, I3_MAT_T(r, c) b, float epsilon) \
    {                                                                                       \
        return !I3_MAT(r, c, eq)(a, b, epsilon);                                            \
    }                                                                                       \
                                                                                            \
    static inline const char* I3_MAT(r, c, dump)(I3_MAT_T(r, c) a)                          \
    {                                                                                       \
        static char buf[256];                                                               \
        int k = 0;                                                                          \
        k += snprintf(buf, sizeof(buf), "[");                                               \
        for (int i = 0; i < r; i++)                                                         \
        {                                                                                   \
            k += snprintf(buf + k, sizeof(buf) - k, "[");                                   \
            for (int j = 0; j < c; j++)                                                     \
            {                                                                               \
                if (j > 0)                                                                  \
                    k += snprintf(buf + k, sizeof(buf) - k, " ");                           \
                k += snprintf(buf + k, sizeof(buf) - k, "%f", a.m[i * c + j]);              \
            }                                                                               \
            k += snprintf(buf + k, sizeof(buf) - k, "]");                                   \
        }                                                                                   \
        k += snprintf(buf + k, sizeof(buf) - k, "]");                                       \
        return buf;                                                                         \
    }

#define I3_MATS_DECL(s)                                   \
    static inline I3_MAT_T(s, s) I3_MAT(s, s, identity)() \
    {                                                     \
        I3_MAT_T(s, s) r;                                 \
        for (int i = 0; i < s; i++)                       \
            for (int j = 0; j < s; j++)                   \
                r.m[i * s + j] = (i == j) ? 1.0f : 0.0f;  \
        return r;                                         \
    }

#define I3_MAT_MUL(x, y, z)                                                                        \
    static inline I3_MAT_T(x, z) I3_MAT(x, y, mul)##_mat##y##z(I3_MAT_T(x, y) a, I3_MAT_T(y, z) b) \
    {                                                                                              \
        I3_MAT_T(x, z) res;                                                                        \
        for (int i = 0; i < x; i++)                                                                \
            for (int j = 0; j < z; j++)                                                            \
            {                                                                                      \
                res.m[i * z + j] = 0.0f;                                                           \
                for (int k = 0; k < y; k++)                                                        \
                    res.m[i * z + j] += a.m[i * y + k] * b.m[k * z + j];                           \
            }                                                                                      \
        return res;                                                                                \
    }

// common matrix types
I3_MAT_STRUCT(2, 2)
I3_MAT_STRUCT(2, 3)
I3_MAT_STRUCT(2, 4)
I3_MAT_STRUCT(3, 2)
I3_MAT_STRUCT(3, 3)
I3_MAT_STRUCT(3, 4)
I3_MAT_STRUCT(4, 2)
I3_MAT_STRUCT(4, 3)
I3_MAT_STRUCT(4, 4)

I3_MAT_DECL(2, 2)
I3_MAT_DECL(2, 3)
I3_MAT_DECL(2, 4)
I3_MAT_DECL(3, 2)
I3_MAT_DECL(3, 3)
I3_MAT_DECL(3, 4)
I3_MAT_DECL(4, 2)
I3_MAT_DECL(4, 3)
I3_MAT_DECL(4, 4)

I3_MATS_DECL(2)
I3_MATS_DECL(3)
I3_MATS_DECL(4)

I3_MAT_MUL(2, 2, 2)
I3_MAT_MUL(3, 3, 3)
I3_MAT_MUL(4, 4, 4)