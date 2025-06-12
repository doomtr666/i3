#pragma once

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

static const float I3_PI = 3.14159265358979323846f;
static const float I3_2_PI = 6.28318530717958647692f;
static const float I3_PI_OVER_2 = 1.57079632679489661923f;
static const float I3_PI_OVER_4 = 0.78539816339744830962f;

// integer function
static inline bool i3_is_odd(int a)
{
    return (a & 1) != 0;
}

static inline bool i3_is_even(int a)
{
    return (a & 1) == 0;
}

// math functions
static inline float i3_absf(float a)
{
    return (a < 0.0f) ? -a : a;
}

static inline float i3_squaref(float a)
{
    return a * a;
}

static inline float i3_sqrtf(float a)
{
    return sqrtf(a);
}

static inline float i3_sinf(float a)
{
    return sinf(a);
}

static inline float i3_cosf(float a)
{
    return cosf(a);
}

static inline float i3_tanf(float a)
{
    return tanf(a);
}

static inline float i3_deg_to_radf(float a)
{
    return a * (I3_PI / 180.0f);
}

static inline float i3_rad_to_degf(float a)
{
    return a * (180.0f / I3_PI);
}

// equal within epsilon
static inline bool i3_eqf(float a, float b, float epsilon)
{
    return i3_absf(a - b) < epsilon;
}

// min max functions
static inline float i3_maxf(float a, float b)
{
    return (a > b) ? a : b;
}

static inline float i3_minf(float a, float b)
{
    return (a < b) ? a : b;
}

static inline float i3_clampf(float a, float min, float max)
{
    return i3_maxf(min, i3_minf(a, max));
}

static inline float i3_saturatef(float a)
{
    return i3_clampf(a, 0.0f, 1.0f);
}
