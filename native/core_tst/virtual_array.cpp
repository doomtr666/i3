#include "gtest/gtest.h"

extern "C"
{
#include "native/core/virtual_array.h"
}

TEST(virtual_array, init_free)
{
    i3_virtual_array_t array;
    i3_virtual_array_init(&array, sizeof(int), 1024);

    EXPECT_EQ(i3_virtual_array_count(&array), 0);
    EXPECT_EQ(i3_virtual_array_capacity(&array), 0);
    EXPECT_EQ(i3_virtual_array_element_size(&array), sizeof(int));
    EXPECT_NE(i3_virtual_array_data(&array), nullptr);

    i3_virtual_array_free(&array);
}

TEST(virtual_array, push)
{
    i3_virtual_array_t array;
    i3_virtual_array_init(&array, sizeof(int), 1024);

    EXPECT_EQ(i3_virtual_array_count(&array), 0);
    EXPECT_EQ(i3_virtual_array_capacity(&array), 0);
    EXPECT_EQ(i3_virtual_array_element_size(&array), sizeof(int));
    EXPECT_NE(i3_virtual_array_data(&array), nullptr);

    int a = 1;
    i3_virtual_array_push(&array, &a);
    EXPECT_EQ(i3_virtual_array_count(&array), 1);
    EXPECT_EQ(i3_virtual_array_capacity(&array), 1);
    EXPECT_EQ(i3_virtual_array_element_size(&array), sizeof(int));
    EXPECT_NE(i3_virtual_array_data(&array), nullptr);
    EXPECT_EQ(*(int*)i3_virtual_array_data(&array), 1);

    int b = 2;
    i3_virtual_array_push(&array, &b);
    EXPECT_EQ(i3_virtual_array_count(&array), 2);
    EXPECT_EQ(i3_virtual_array_capacity(&array), 2);
    EXPECT_EQ(i3_virtual_array_element_size(&array), sizeof(int));
    EXPECT_NE(i3_virtual_array_data(&array), nullptr);
    EXPECT_EQ(*(int*)i3_virtual_array_data(&array), 1);
    EXPECT_EQ(*(int*)i3_virtual_array_at(&array, 0), 1);
    EXPECT_EQ(*(int*)i3_virtual_array_at(&array, 1), 2);

    i3_virtual_array_free(&array);
}

TEST(virtual_array, addn)
{
    i3_virtual_array_t array;
    i3_virtual_array_init(&array, sizeof(int), 1024);

    EXPECT_EQ(i3_virtual_array_count(&array), 0);
    EXPECT_EQ(i3_virtual_array_capacity(&array), 0);
    EXPECT_EQ(i3_virtual_array_element_size(&array), sizeof(int));
    EXPECT_NE(i3_virtual_array_data(&array), nullptr);

    int a = 1;
    int b = 2;
    int c = 3;
    int d = 4;
    int e = 5;
    int f = 6;
    int g = 7;
    int h = 8;
    int i = 9;
    int j = 10;

    int* data = (int*)i3_virtual_array_addn(&array, 10);
    data[0] = a;
    data[1] = b;
    data[2] = c;
    data[3] = d;
    data[4] = e;
    data[5] = f;
    data[6] = g;
    data[7] = h;
    data[8] = i;
    data[9] = j;

    EXPECT_EQ(i3_virtual_array_count(&array), 10);
    EXPECT_EQ(i3_virtual_array_capacity(&array), 16);
    EXPECT_EQ(i3_virtual_array_element_size(&array), sizeof(int));
    EXPECT_NE(i3_virtual_array_data(&array), nullptr);
    EXPECT_EQ(*(int*)i3_virtual_array_data(&array), 1);
    EXPECT_EQ(*(int*)i3_virtual_array_at(&array, 0), 1);
    EXPECT_EQ(*(int*)i3_virtual_array_at(&array, 1), 2);
    EXPECT_EQ(*(int*)i3_virtual_array_at(&array, 2), 3);
    EXPECT_EQ(*(int*)i3_virtual_array_at(&array, 3), 4);
    EXPECT_EQ(*(int*)i3_virtual_array_at(&array, 4), 5);
    EXPECT_EQ(*(int*)i3_virtual_array_at(&array, 5), 6);
    EXPECT_EQ(*(int*)i3_virtual_array_at(&array, 6), 7);
    EXPECT_EQ(*(int*)i3_virtual_array_at(&array, 7), 8);
    EXPECT_EQ(*(int*)i3_virtual_array_at(&array, 8), 9);
    EXPECT_EQ(*(int*)i3_virtual_array_at(&array, 9), 10);

    i3_virtual_array_free(&array);
}

TEST(virtual_array, front_back_at)
{
    i3_virtual_array_t array;
    i3_virtual_array_init(&array, sizeof(int), 1024);

    EXPECT_EQ(i3_virtual_array_count(&array), 0);
    EXPECT_EQ(i3_virtual_array_capacity(&array), 0);
    EXPECT_EQ(i3_virtual_array_element_size(&array), sizeof(int));
    EXPECT_NE(i3_virtual_array_data(&array), nullptr);

    int* data = (int*)i3_virtual_array_addn(&array, 5);
    data[0] = 1;
    data[1] = 2;
    data[2] = 3;
    data[3] = 4;
    data[4] = 5;

    EXPECT_EQ(i3_virtual_array_count(&array), 5);
    EXPECT_EQ(i3_virtual_array_capacity(&array), 8);
    EXPECT_EQ(i3_virtual_array_element_size(&array), sizeof(int));
    EXPECT_NE(i3_virtual_array_data(&array), nullptr);

    EXPECT_EQ(*(int*)i3_virtual_array_data(&array), 1);

    EXPECT_EQ(*(int*)i3_virtual_array_front(&array), 1);
    EXPECT_EQ(*(int*)i3_virtual_array_back(&array), 5);

    EXPECT_EQ(*(int*)i3_virtual_array_at(&array, 0), 1);
    EXPECT_EQ(*(int*)i3_virtual_array_at(&array, 1), 2);
    EXPECT_EQ(*(int*)i3_virtual_array_at(&array, 2), 3);
    EXPECT_EQ(*(int*)i3_virtual_array_at(&array, 3), 4);
    EXPECT_EQ(*(int*)i3_virtual_array_at(&array, 4), 5);

    i3_virtual_array_free(&array);
}

TEST(virtual_array, stress)
{
    const uint32_t count = 1000000;

    i3_virtual_array_t array;
    i3_virtual_array_init(&array, sizeof(int), I3_GB);

    for (uint32_t i = 0; i < count; i++)
    {
        i3_virtual_array_push(&array, &i);
    }

    EXPECT_EQ(i3_virtual_array_count(&array), count);

    for (uint32_t i = 0; i < count; i++)
    {
        EXPECT_EQ(*(int*)i3_virtual_array_at(&array, i), i);
    }

    i3_virtual_array_free(&array);
}

TEST(virtual_array, struct)
{
    typedef struct element_t
    {
        uint32_t a;
        uint32_t b;
        uint32_t c;
    } element_t;

    i3_virtual_array_t array;
    i3_virtual_array_init(&array, sizeof(element_t), 1024);

    EXPECT_EQ(i3_virtual_array_count(&array), 0);
    EXPECT_EQ(i3_virtual_array_capacity(&array), 0);
    EXPECT_EQ(i3_virtual_array_element_size(&array), sizeof(element_t));
    EXPECT_NE(i3_virtual_array_data(&array), nullptr);

    for (uint32_t i = 0; i < 1024; i++)
    {
        element_t element = { i, i + 1, i + 2 };
        i3_virtual_array_push(&array, &element);
    }

    EXPECT_EQ(i3_virtual_array_count(&array), 1024);

    for (uint32_t i = 0; i < 1024; i++)
    {
        element_t* element = (element_t*)i3_virtual_array_at(&array, i);
        EXPECT_EQ(element->a, i);
        EXPECT_EQ(element->b, i + 1);
        EXPECT_EQ(element->c, i + 2);
    }

    i3_virtual_array_free(&array);
}