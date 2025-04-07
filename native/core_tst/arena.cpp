#include "gtest/gtest.h"

extern "C"
{
#include "native/core/arena.h"
}

TEST(arena, init_free)
{
    i3_arena_t arena;
    i3_arena_init(&arena, 1024);
    i3_arena_free(&arena);
}

TEST(arena, alloc_multiple)
{
    i3_arena_t arena;
    i3_arena_init(&arena, 64);

    // one block needed
    EXPECT_NE(i3_arena_alloc(&arena, 16), nullptr);
    EXPECT_NE(i3_arena_alloc(&arena, 16), nullptr);
    EXPECT_NE(i3_arena_alloc(&arena, 16), nullptr);
    EXPECT_NE(i3_arena_alloc(&arena, 16), nullptr);
    EXPECT_EQ(i3_arena_allocated(&arena), 64);
    EXPECT_EQ(i3_arena_allocation_count(&arena), 1);

    // second block
    EXPECT_NE(i3_arena_alloc(&arena, 16), nullptr);
    EXPECT_EQ(i3_arena_allocated(&arena), 128);
    EXPECT_EQ(i3_arena_allocation_count(&arena), 2);

    i3_arena_free(&arena);
}

TEST(arena, alloc_large)
{
    i3_arena_t arena;
    i3_arena_init(&arena, 1024);
    i3_arena_alloc(&arena, 1);
    i3_arena_alloc(&arena, 2);
    i3_arena_alloc(&arena, 3);
    i3_arena_alloc(&arena, 4);
    i3_arena_alloc(&arena, 5);
    i3_arena_alloc(&arena, 6);
    i3_arena_alloc(&arena, 7);
    i3_arena_alloc(&arena, 8);
    i3_arena_alloc(&arena, 9);
    i3_arena_alloc(&arena, 10);

    void* ptr = i3_arena_alloc(&arena, 788);
    EXPECT_NE(ptr, nullptr);

    EXPECT_EQ(i3_arena_allocated(&arena), 1024 + 788);
    EXPECT_EQ(i3_arena_allocation_count(&arena), 2);

    i3_arena_free(&arena);
}