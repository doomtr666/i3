
extern "C"
{
#include "common.h"
}

#define VMA_IMPLEMENTATION

// to debug vma leaks
#if 0
    #define VMA_DEBUG_LOG_FORMAT(format, ...)                              \
        do                                                                 \
        {                                                                  \
            i3_log_err(i3_vk_get_logger(), "VMA: " format, ##__VA_ARGS__); \
        } while (false)
#endif

#include "vk_mem_alloc.h"
