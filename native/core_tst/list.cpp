#include "gtest/gtest.h"

extern "C"
{
#include "native/core/list.h"  
}

typedef struct dnode
{
    struct dnode* prev;
    struct dnode* next;
    int value;
} dnode;

TEST(list, init)
{
    i3_dlist(dnode) head;
    i3_dlist_init(&head);

    EXPECT_EQ(head.first, nullptr);
    EXPECT_EQ(head.last, nullptr);
    EXPECT_TRUE(i3_dlist_empty(&head));
}

TEST(list, append)
{
    i3_dlist(dnode) head;
    i3_dlist_init(&head);

    struct dnode node1 = { nullptr, nullptr, 1 };
    i3_dlist_append(&head, &node1);
    EXPECT_EQ(head.first, &node1);
    EXPECT_EQ(head.last, &node1);
    EXPECT_FALSE(i3_dlist_empty(&head));

    struct dnode node2 = { nullptr, nullptr, 2 };
    i3_dlist_append(&head, &node2);
    EXPECT_EQ(head.first, &node1);
    EXPECT_EQ(head.last, &node2);
    EXPECT_FALSE(i3_dlist_empty(&head));

    struct dnode node3 = { nullptr, nullptr, 3 };
    i3_dlist_append(&head, &node3);
    EXPECT_EQ(head.first, &node1);
    EXPECT_EQ(head.last, &node3);
    EXPECT_FALSE(i3_dlist_empty(&head));
}

TEST(list, preprend)
{
    i3_dlist(dnode) head;
    i3_dlist_init(&head);

    struct dnode node1 = { nullptr, nullptr, 1 };
    i3_dlist_prepend(&head, &node1);
    EXPECT_EQ(head.first, &node1);
    EXPECT_EQ(head.last, &node1);
    EXPECT_FALSE(i3_dlist_empty(&head));

    struct dnode node2 = { nullptr, nullptr, 2 };
    i3_dlist_prepend(&head, &node2);
    EXPECT_EQ(head.first, &node2);
    EXPECT_EQ(head.last, &node1);
    EXPECT_FALSE(i3_dlist_empty(&head));

    struct dnode node3 = { nullptr, nullptr, 3 };
    i3_dlist_prepend(&head, &node3);
    EXPECT_EQ(head.first, &node3);
    EXPECT_EQ(head.last, &node1);
    EXPECT_FALSE(i3_dlist_empty(&head));
}

TEST(list, remove)
{
    i3_dlist(dnode) head;
    i3_dlist_init(&head);
    struct dnode node1 = { nullptr, nullptr, 1 };
    i3_dlist_append(&head, &node1);
    struct dnode node2 = { nullptr, nullptr, 2 };
    i3_dlist_append(&head, &node2);
    struct dnode node3 = { nullptr, nullptr, 3 };
    i3_dlist_append(&head, &node3);


    i3_dlist_remove(&head, &node2);
    EXPECT_EQ(head.first, &node1);
    EXPECT_EQ(head.last, &node3);
    EXPECT_FALSE(i3_dlist_empty(&head));

    i3_dlist_remove(&head, &node1);
    EXPECT_EQ(head.first, &node3);
    EXPECT_EQ(head.last, &node3);
    EXPECT_FALSE(i3_dlist_empty(&head));

    i3_dlist_remove(&head, &node3);
    EXPECT_EQ(head.first, nullptr);
    EXPECT_EQ(head.last, nullptr);
    EXPECT_TRUE(i3_dlist_empty(&head));
}

TEST(list, copy)
{
    i3_dlist(dnode) head1;
    i3_dlist_init(&head1);
    struct dnode node1 = { nullptr, nullptr, 1 };
    i3_dlist_append(&head1, &node1);
    struct dnode node2 = { nullptr, nullptr, 2 };
    i3_dlist_append(&head1, &node2);
    struct dnode node3 = { nullptr, nullptr, 3 };
    i3_dlist_append(&head1, &node3);

    i3_dlist(dnode) head2;
    i3_dlist_copy(&head2, &head1);

    EXPECT_EQ(head2.first, head1.first);
    EXPECT_EQ(head2.last, head1.last);
    EXPECT_FALSE(i3_dlist_empty(&head2));
}

TEST(list, accessors)
{
    i3_dlist(dnode) head;
    i3_dlist_init(&head);
    struct dnode node1 = { nullptr, nullptr, 1 };
    i3_dlist_append(&head, &node1);
    struct dnode node2 = { nullptr, nullptr, 2 };
    i3_dlist_append(&head, &node2);
    struct dnode node3 = { nullptr, nullptr, 3 };
    i3_dlist_append(&head, &node3);

    EXPECT_EQ(i3_dlist_first(&head), &node1);
    EXPECT_EQ(i3_dlist_last(&head), &node3);
    EXPECT_EQ(i3_dlist_next(&node1), &node2);
    EXPECT_EQ(i3_dlist_prev(&node2), &node1);
}

TEST(list, foreach)
{
    i3_dlist(dnode) head;
    i3_dlist_init(&head);
    struct dnode node1 = { nullptr, nullptr, 1 };
    i3_dlist_append(&head, &node1);
    struct dnode node2 = { nullptr, nullptr, 2 };
    i3_dlist_append(&head, &node2);
    struct dnode node3 = { nullptr, nullptr, 3 };
    i3_dlist_append(&head, &node3);

    int sum = 0;
    struct dnode* node;
    i3_dlist_foreach(&head, node)
    {
        sum += node->value;
    }

    EXPECT_EQ(sum, 6);
}

TEST(list, foreach_r)
{
    i3_dlist(dnode) head;
    i3_dlist_init(&head);
    struct dnode node1 = { nullptr, nullptr, 1 };
    i3_dlist_append(&head, &node1);
    struct dnode node2 = { nullptr, nullptr, 2 };
    i3_dlist_append(&head, &node2);
    struct dnode node3 = { nullptr, nullptr, 3 };
    i3_dlist_append(&head, &node3);

    int sum = 0;
    struct dnode* node;
    i3_dlist_foreach_r(&head, node)
    {
        sum += node->value;
    }

    EXPECT_EQ(sum, 6);
}