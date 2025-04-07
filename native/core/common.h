#pragma once

#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>

// TODO: Add more platform support
#define I3_PLATFORM_UNKNOWN 0
#define I3_PLATFORM_WINDOWS 1

#ifdef _WIN32
#define I3_PLATFORM I3_PLATFORM_WINDOWS
#include <intrin.h>
#else
#define I3_PLATFORM I3_PLATFORM_UNKNOWN
#endif

// debug flag
#ifdef _DEBUG
#define I3_DEBUG 1
#endif

// dll export
#if I3_PLATFORM == I3_PLATFORM_WINDOWS
#define I3_EXPORT __declspec(dllexport)
#else
#define I3_EXPORT
#endif

// compatibility
#if I3_PLATFORM == I3_PLATFORM_WINDOWS
#define strdup _strdup
#define i3_stackalloc _malloca
#define i3_alignof(x) __alignof(x)
#define i3_break() __debugbreak()
#endif

// dbg allocator
I3_EXPORT void* i3_dbg_alloc_(size_t size, const char* file, int line);
I3_EXPORT void* i3_dbg_calloc_(size_t n, size_t size, const char* file, int line);
I3_EXPORT void* i3_dbg_realloc_(void* ptr, size_t size, const char* file, int line);
I3_EXPORT void i3_dbg_free_(void* ptr, const char* file, int line);

// dbg memory leaks
I3_EXPORT void i3_dbg_break_on_alloc(uint32_t index);

// allocation
#ifdef I3_DEBUG
#define i3_alloc(size) i3_dbg_alloc_(size, __FILE__, __LINE__)
#define i3_calloc(count, size) i3_dbg_calloc_(count, size, __FILE__, __LINE__)
#define i3_zalloc(size) i3_dbg_calloc_(1, size, __FILE__, __LINE__)
#define i3_realloc(ptr, size) i3_dbg_realloc_(ptr, size, __FILE__, __LINE__)
#define i3_free(ptr) i3_dbg_free_(ptr, __FILE__, __LINE__)
#else
#define i3_alloc(size) malloc(size)
#define i3_calloc(count, size) calloc(count, size)
#define i3_zalloc(size) i3_calloc(1,size)
#define i3_realloc(ptr, size) realloc(ptr, size)
#define i3_free(ptr) free(ptr)
#endif

// default alignment
#define I3_DEFAULT_ALIGN sizeof(void *)

// check if a is a correct alignment, positive power of two
#define i3_check_align(a) (((a) > 0) && (((a) & ((a)-1)) ==0))

// align value
#define i3_align_v(v, align) (((v) + ((align)-1)) & (~((align)-1)))

// align ptr
#define i3_align_p(p, align) ((void *)(i3_align_v((uintptr_t)(p), (uintptr_t)(align))))

// sizes
#define I3_KB 1024ULL
#define I3_MB (1024ULL * I3_KB)
#define I3_GB (1024ULL * I3_MB)
#define I3_TB (1024ULL * I3_GB)
#define I3_PAGE_SIZE (4ULL * I3_KB)

// flag
#define i3_flag(x) (1 << (x))

// min, max, clamp
#define i3_min(a, b) ((a) < (b) ? (a) : (b))
#define i3_max(a, b) ((a) > (b) ? (a) : (b))
#define i3_clamp(v, min, max) i3_min(i3_max(v, min), max)