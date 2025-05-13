#include "time.h"

#if I3_PLATFORM == I3_PLATFORM_WINDOWS
    #define WIN32_MEAN_AND_LEAN
    #include <windows.h>

i3_hr_time_t i3_hr_time_now()
{
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return counter.QuadPart;
}

double i3_hr_time_to_sec(i3_hr_time_t time)
{
    static LARGE_INTEGER frequency;
    if (frequency.QuadPart == 0)
        QueryPerformanceFrequency(&frequency);
    return time / (double)frequency.QuadPart;
}

#endif