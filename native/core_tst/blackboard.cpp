#include "gtest/gtest.h"

extern "C"
{
#include "native/core/blackboard.h"
}

TEST(blackboard, init_destroy)
{
    i3_blackboard_t blackboard;
    i3_blackboard_init(&blackboard);
    i3_blackboard_destroy(&blackboard);
}

TEST(blackboard, put_get)
{
    i3_blackboard_t blackboard;
    i3_blackboard_init(&blackboard);

    int v1 = 42;
    int v2 = 43;
    int v3 = 44;

    ASSERT_TRUE(i3_blackboard_put(&blackboard, "V1", &v1, sizeof(v1)));
    ASSERT_TRUE(i3_blackboard_put(&blackboard, "V2", &v2, sizeof(v1)));
    ASSERT_TRUE(i3_blackboard_put(&blackboard, "V3", &v3, sizeof(v1)));

    int v1_out = 0;
    int v2_out = 0;
    int v3_out = 0;

    ASSERT_TRUE(i3_blackboard_get(&blackboard, "V1", &v1_out));
    ASSERT_TRUE(i3_blackboard_get(&blackboard, "V2", &v2_out));
    ASSERT_TRUE(i3_blackboard_get(&blackboard, "V3", &v3_out));

    ASSERT_EQ(v1, v1_out);
    ASSERT_EQ(v2, v2_out);
    ASSERT_EQ(v3, v3_out);

    i3_blackboard_destroy(&blackboard);
}