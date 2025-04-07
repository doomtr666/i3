#include "gtest/gtest.h"

extern "C"
{
#include "native/core/memory_pool.h"
}

TEST(memory_pool, init_destroy)
{
    i3_memory_pool_t pool;
    i3_memory_pool_init(&pool, 0, sizeof(int), 16);
    EXPECT_EQ(i3_memory_pool_total_capacity(&pool), 0);
    EXPECT_EQ(i3_memory_pool_allocated(&pool), 0);
    i3_memory_pool_destroy(&pool);
}

TEST(memory_pool, alloc_free)
{
    i3_memory_pool_t pool;
    i3_memory_pool_init(&pool, 0, sizeof(int), 16);
    EXPECT_EQ(i3_memory_pool_total_capacity(&pool), 0);
    EXPECT_EQ(i3_memory_pool_allocated(&pool), 0);

    int* a = (int*)i3_memory_pool_alloc(&pool);
    EXPECT_EQ(i3_memory_pool_total_capacity(&pool), 16);
    EXPECT_EQ(i3_memory_pool_allocated(&pool), 1);
    *a = 1;

    int* b = (int*)i3_memory_pool_alloc(&pool);
    EXPECT_EQ(i3_memory_pool_total_capacity(&pool), 16);
    EXPECT_EQ(i3_memory_pool_allocated(&pool), 2);
    *b = 2;

    i3_memory_pool_free(&pool, a);
    EXPECT_EQ(i3_memory_pool_total_capacity(&pool), 16);
    EXPECT_EQ(i3_memory_pool_allocated(&pool), 1);

    i3_memory_pool_free(&pool, b);
    EXPECT_EQ(i3_memory_pool_total_capacity(&pool), 16);
    EXPECT_EQ(i3_memory_pool_allocated(&pool), 0);

    i3_memory_pool_destroy(&pool);
}

TEST(memory_pool, recycle)
{
    i3_memory_pool_t pool;
    i3_memory_pool_init(&pool, 0, sizeof(int), 16);

    int* a = (int*)i3_memory_pool_alloc(&pool);
    int* b = (int*)i3_memory_pool_alloc(&pool);
    i3_memory_pool_free(&pool, a);

    int* c = (int*)i3_memory_pool_alloc(&pool);

    // check if a is recycled
    EXPECT_EQ(a, c);

    i3_memory_pool_free(&pool, b);

    int* d = (int*)i3_memory_pool_alloc(&pool);

    // check if b is recycled
    EXPECT_EQ(b, d);

    i3_memory_pool_destroy(&pool);
}

TEST(memory_pool, align)
{
    i3_memory_pool_t pool;
    i3_memory_pool_init(&pool, 16, sizeof(int), 16);
    int* a = (int*)i3_memory_pool_alloc(&pool);
    EXPECT_EQ((uintptr_t)a % 16, 0);

    int* b = (int*)i3_memory_pool_alloc(&pool);
    EXPECT_EQ((uintptr_t)b % 16, 0);

    int* c = (int*)i3_memory_pool_alloc(&pool);
    EXPECT_EQ((uintptr_t)c % 16, 0);

    i3_memory_pool_destroy(&pool);
}

TEST(memory_pool, multiple_pages)
{
    const int count = 1024;

    i3_memory_pool_t pool;
    i3_memory_pool_init(&pool, 0, sizeof(int), 16);

    int* a[count];

    for (int i = 0; i < count; ++i)
    {
        a[i] = (int*)i3_memory_pool_alloc(&pool);
        *a[i] = i;
    }

    // free half
    for (int i = 0; i < count / 2; ++i)
        i3_memory_pool_free(&pool, a[i]);

    // alloc again
    for (int i = 0; i < count / 2; ++i)
    {
        a[i] = (int*)i3_memory_pool_alloc(&pool);
        *a[i] = i;
    }

    // check values
    for (int i = 0; i < count; ++i)
        EXPECT_EQ(*a[i], i);

    // check capacity and count
    EXPECT_EQ(i3_memory_pool_total_capacity(&pool), 1024);
    EXPECT_EQ(i3_memory_pool_allocated(&pool), count);

    i3_memory_pool_destroy(&pool);
}