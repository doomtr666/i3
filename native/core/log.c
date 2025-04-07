#include "log.h"

#include "hashtable.h"
#include "list.h"

#include <stdarg.h>
#include <stdio.h>

struct i3_logger_o
{
    i3_logger_i iface;
    const char* name;
    i3_log_level_t level;
    i3_logger_o* prev;
    i3_logger_o* next;
};

typedef struct
{
    i3_hashtable_t loggers_table;
    i3_dlist(i3_logger_o) loggers_list;
} i3_loggers_t;

static i3_loggers_t* i3_loggers_;

static const char* i3_log_level_to_string(i3_log_level_t level)
{
    switch (level)
    {
    case I3_LOG_LEVEL_DEBUG:
        return "DBG";
    case I3_LOG_LEVEL_INFO:
        return "INF";
    case I3_LOG_LEVEL_WARN:
        return "WRN";
    case I3_LOG_LEVEL_ERROR:
        return "ERR";
    default:
        return "UNK";
    }
}

static const char* i3_get_color_from_level(i3_log_level_t level)
{
    switch (level)
    {
    case I3_LOG_LEVEL_DEBUG:
        return "\033[32m";
    case I3_LOG_LEVEL_INFO:
        return "\033[37m";
    case I3_LOG_LEVEL_WARN:
        return "\033[33m\033[1m";
    case I3_LOG_LEVEL_ERROR:
        return "\033[31m\033[1m";
    default:
        return "";
    }
}

static void i3_logger_set_level(i3_logger_o* self, i3_log_level_t level)
{
    assert(self != NULL);

    self->level = level;
}

static void i3_logger_log(i3_logger_o* self, i3_log_level_t level, const char* format, ...)
{
    assert(self != NULL);
    assert(format != NULL);

    if (level < self->level)
        return;

    va_list args;
    va_start(args, format);

    // log
    static const char* reset_colors = "\033[m";

    // log if enabled
    if (level >= self->level)
    {
        fprintf(stdout, "%s%s - %s - ", i3_get_color_from_level(level), i3_log_level_to_string(level), self->name);
        vfprintf(stdout, format, args);
        fprintf(stdout, "%s\n", reset_colors);
        fflush(stdout);
    }

    va_end(args);
}

static void i3_destroy_loggers(void)
{
    // destroy loggers
    while (!i3_dlist_empty(&i3_loggers_->loggers_list))
    {
        i3_logger_o* last = i3_dlist_last(&i3_loggers_->loggers_list);
        i3_dlist_remove(&i3_loggers_->loggers_list, last);
        i3_free(last);
    }

    // destroy hashtable
    i3_hashtable_free(&i3_loggers_->loggers_table);

    i3_free(i3_loggers_);
    i3_loggers_ = NULL;
}

static i3_logger_i logger_iface__ = {
    .self = NULL,
    .set_level = i3_logger_set_level,
    .log = i3_logger_log
};

i3_logger_i* i3_get_logger(const char* name)
{
    assert(name != NULL);

    if (i3_loggers_ == NULL)
    {
        i3_loggers_ = i3_alloc(sizeof(i3_loggers_t));
        i3_dlist_init(&i3_loggers_->loggers_list);
        i3_hashtable_init(&i3_loggers_->loggers_table);
        atexit(i3_destroy_loggers);
    }

    uint32_t namelen = (uint32_t)strlen(name);

    // return logger if it already exists   
    i3_logger_o* logger = (i3_logger_o*)i3_hashtable_find(&i3_loggers_->loggers_table, name, namelen);
    if (logger != NULL)
        return &logger->iface;

    // create logger
    logger = i3_alloc(sizeof(i3_logger_o));
    assert(logger != NULL);

    logger->iface = logger_iface__;
    logger->iface.self = logger;
    logger->name = name; // I consider name is a literal string here, no need to copy
    logger->level = I3_LOG_LEVEL_INFO;
    logger->prev = logger->next = NULL;

    // insert to hashtable
    i3_hashtable_insert(&i3_loggers_->loggers_table, logger->name, namelen, logger);

    // insert to logger list
    i3_dlist_append(&i3_loggers_->loggers_list, logger);

    return &logger->iface;
}