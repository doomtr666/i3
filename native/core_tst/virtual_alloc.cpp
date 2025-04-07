#include "gtest/gtest.h"

extern "C"
{
#include "native/core/virtual_alloc.h"
}

TEST(virtual_alloc, alloc_free)
{
    void* ptr = i3_virtual_alloc(I3_GB);
    EXPECT_NE(ptr, nullptr);

    EXPECT_DEATH(*(uint32_t*)ptr = 42, "");

    EXPECT_TRUE(i3_virtual_free(ptr));
}

TEST(virtual_alloc, commit_decommit)
{
    void* ptr = i3_virtual_alloc(I3_GB);
    EXPECT_NE(ptr, nullptr);
    EXPECT_DEATH(*(uint32_t*)ptr = 42, "");

    EXPECT_TRUE(i3_virtual_commit(ptr, 4 * I3_KB));
    *(uint32_t*)ptr = 42;
    EXPECT_TRUE(i3_virtual_decommit(ptr, 4 * I3_KB));
    EXPECT_DEATH(*(uint32_t*)ptr = 42, "");

    EXPECT_TRUE(i3_virtual_free(ptr));
}

TEST(virtual_alloc, alloc_large)
{
    void* ptr = i3_virtual_alloc(64ULL * I3_TB);
    EXPECT_NE(ptr, nullptr);

    EXPECT_TRUE(i3_virtual_free(ptr));
}