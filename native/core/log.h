#pragma once

#include "common.h"

typedef struct i3_logger_i i3_logger_i;
typedef struct i3_logger_o i3_logger_o;

typedef enum i3_log_level_t
{
    I3_LOG_LEVEL_ALL = 0,
    I3_LOG_LEVEL_DEBUG = 1,
    I3_LOG_LEVEL_INFO = 2,
    I3_LOG_LEVEL_WARN = 3,
    I3_LOG_LEVEL_ERROR = 4,
    I3_LOG_LEVEL_DISABLE = UINT32_MAX,
} i3_log_level_t;

struct i3_logger_i
{
    i3_logger_o* self;

    void (*set_level)(i3_logger_o* self, i3_log_level_t level);
    void (*log)(i3_logger_o* self, i3_log_level_t level, const char* format, ...);
};

I3_EXPORT i3_logger_i* i3_get_logger(const char* name);

#ifdef I3_DEBUG
    #define i3_log_dbg(logger, ...) (logger)->log((logger)->self, I3_LOG_LEVEL_DEBUG, __VA_ARGS__)
#else
    #define i3_log_dbg(logger, ...)
#endif
#define i3_log_inf(logger, ...) (logger)->log((logger)->self, I3_LOG_LEVEL_INFO, __VA_ARGS__)
#define i3_log_wrn(logger, ...) (logger)->log((logger)->self, I3_LOG_LEVEL_WARN, __VA_ARGS__)
#define i3_log_err(logger, ...) (logger)->log((logger)->self, I3_LOG_LEVEL_ERROR, __VA_ARGS__)
