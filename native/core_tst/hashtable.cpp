#include "gtest/gtest.h"

extern "C"
{
#include "native/core/hashtable.h"
}

TEST(hashtable, create_destroy)
{
    i3_hashtable_t ht;
    i3_hashtable_init(&ht);
    EXPECT_EQ(i3_hashtable_count(&ht), 0);
    i3_hashtable_free(&ht);
}

TEST(hashtable, insert_find_remove)
{
    i3_hashtable_t ht;
    i3_hashtable_init(&ht);

    int a = 1;
    int b = 2;
    int c = 3;

    i3_hashtable_insert(&ht, &a, sizeof(int), &a);
    i3_hashtable_insert(&ht, &b, sizeof(int), &b);
    i3_hashtable_insert(&ht, &c, sizeof(int), &c);

    EXPECT_EQ(*(int*)i3_hashtable_find(&ht, &a, sizeof(int)), a);
    EXPECT_EQ(*(int*)i3_hashtable_find(&ht, &b, sizeof(int)), b);
    EXPECT_EQ(*(int*)i3_hashtable_find(&ht, &c, sizeof(int)), c);

    i3_hashtable_remove(&ht, &a, sizeof(int));
    EXPECT_EQ(i3_hashtable_find(&ht, &a, sizeof(int)), nullptr);

    // check if other keys are still present
    EXPECT_EQ(*(int*)i3_hashtable_find(&ht, &b, sizeof(int)), b);
    EXPECT_EQ(*(int*)i3_hashtable_find(&ht, &c, sizeof(int)), c);

    i3_hashtable_remove(&ht, &b, sizeof(int));
    EXPECT_EQ(i3_hashtable_find(&ht, &b, sizeof(int)), nullptr);

    // check if other keys are still present
    EXPECT_EQ(*(int*)i3_hashtable_find(&ht, &c, sizeof(int)), c);

    i3_hashtable_remove(&ht, &c, sizeof(int));
    EXPECT_EQ(i3_hashtable_find(&ht, &c, sizeof(int)), nullptr);

    i3_hashtable_free(&ht);
}

TEST(hashtable, clear)
{
    i3_hashtable_t ht;
    i3_hashtable_init(&ht);

    int a = 1;
    int b = 2;
    int c = 3;

    i3_hashtable_insert(&ht, &a, sizeof(int), &a);
    i3_hashtable_insert(&ht, &b, sizeof(int), &b);
    i3_hashtable_insert(&ht, &c, sizeof(int), &c);

    EXPECT_EQ(*(int*)i3_hashtable_find(&ht, &a, sizeof(int)), a);
    EXPECT_EQ(*(int*)i3_hashtable_find(&ht, &b, sizeof(int)), b);
    EXPECT_EQ(*(int*)i3_hashtable_find(&ht, &c, sizeof(int)), c);

    i3_hashtable_clear(&ht);

    EXPECT_EQ(i3_hashtable_find(&ht, &a, sizeof(int)), nullptr);
    EXPECT_EQ(i3_hashtable_find(&ht, &b, sizeof(int)), nullptr);
    EXPECT_EQ(i3_hashtable_find(&ht, &c, sizeof(int)), nullptr);

    i3_hashtable_free(&ht);
}

TEST(hashtable, count)
{
    i3_hashtable_t ht;
    i3_hashtable_init(&ht);

    int a = 1;
    int b = 2;
    int c = 3;

    i3_hashtable_insert(&ht, &a, sizeof(int), &a);
    i3_hashtable_insert(&ht, &b, sizeof(int), &b);
    i3_hashtable_insert(&ht, &c, sizeof(int), &c);

    EXPECT_EQ(i3_hashtable_count(&ht), 3);

    i3_hashtable_remove(&ht, &a, sizeof(int));
    EXPECT_EQ(i3_hashtable_count(&ht), 2);

    i3_hashtable_remove(&ht, &b, sizeof(int));
    EXPECT_EQ(i3_hashtable_count(&ht), 1);

    // remove non-existent key
    i3_hashtable_remove(&ht, &a, sizeof(int));
    EXPECT_EQ(i3_hashtable_count(&ht), 1);

    // remove last key
    i3_hashtable_remove(&ht, &c, sizeof(int));
    EXPECT_EQ(i3_hashtable_count(&ht), 0);

    i3_hashtable_free(&ht);
}

// insert find large number of elements to force hashtable to grow
TEST(hashtable, insert_find_large)
{
    const int count = 10000;

    int* values = new int[count];
    for (uint32_t i = 0; i < count; i++)
        values[i] = i;

    i3_hashtable_t ht;
    i3_hashtable_init(&ht);

    for (int i = 0; i < count; i++)
    {
        i3_hashtable_insert(&ht, &values[i], sizeof(int), &values[i]);
    }

    for (int i = 0; i < count; i++)
    {
        EXPECT_EQ(*(int*)i3_hashtable_find(&ht, &values[i], sizeof(int)), values[i]);
    }

    i3_hashtable_free(&ht);

    delete[] values;
}