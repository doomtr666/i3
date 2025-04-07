#include "gtest/gtest.h"

extern "C"
{
#include "native/core/hash.h"
}

TEST(hash, hash32)
{
    // from https://murmurhash.shorelabs.com/
    EXPECT_EQ(i3_hash32("a", 1, 0), 1009084850);
    EXPECT_EQ(i3_hash32("hello", 5, 0), 613153351);
    EXPECT_EQ(i3_hash32("the ultimate answer is 42.", 26, 0), 146687013);
}
