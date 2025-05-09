#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdio.h>

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

static inline float i3_sqrtf(float a)
{
    return sqrtf(a);
}

// equal wih in epsilon
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
