#pragma once

#include "common.h"

typedef uint64_t i3_hr_time_t;

// high resolution time, not consistent across platforms
i3_hr_time_t i3_hr_time_now();
double i3_hr_time_to_sec(i3_hr_time_t time);

// stopwatch
typedef struct
{
    i3_hr_time_t start;
    i3_hr_time_t end;
} i3_stopwatch_t;

static inline void i3_init_stopwatch(i3_stopwatch_t* sw)
{
    assert(sw != NULL);

    sw->end = 0;
    sw->start = i3_hr_time_now();
}

static inline void i3_stopwatch_start(i3_stopwatch_t* sw)
{
    assert(sw != NULL);

    sw->start = i3_hr_time_now();
}

static inline void i3_stopwatch_stop(i3_stopwatch_t* sw)
{
    assert(sw != NULL);

    sw->end = i3_hr_time_now();
}

static inline double i3_stopwatch_elapsed(i3_stopwatch_t* sw)
{
    assert(sw != NULL);

    return i3_hr_time_to_sec(sw->end - sw->start);
}

static inline double i3_stopwatch_elapsed_ms(i3_stopwatch_t* sw)
{
    assert(sw != NULL);

    return 1000.0 * i3_stopwatch_elapsed(sw);
}

// game time
typedef struct
{
    // frame time
    double total_time;
    double elapsed_time;
    uint32_t frame_count;
    // timers
    i3_hr_time_t start_time;
    i3_hr_time_t last_time;
} i3_game_time_t;

static inline void i3_game_time_init(i3_game_time_t* game_time)
{
    assert(game_time != NULL);

    memset(game_time, 0, sizeof(i3_game_time_t));

    i3_hr_time_t now = i3_hr_time_now();
    game_time->start_time = now;
    game_time->last_time = now;
}

static inline void i3_game_time_update(i3_game_time_t* game_time)
{
    assert(game_time != NULL);

    i3_hr_time_t now = i3_hr_time_now();

    game_time->elapsed_time = i3_hr_time_to_sec(now - game_time->last_time);
    game_time->total_time = i3_hr_time_to_sec(now - game_time->start_time);
    game_time->last_time = now;
    game_time->frame_count++;
}