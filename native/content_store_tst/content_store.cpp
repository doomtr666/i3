#include <gtest/gtest.h>

extern "C"
{
#include "native/content_store/content_store.h"
}

TEST(content_store, create_and_destroy)
{
    i3_content_store_i* store = i3_content_store_create();
    ASSERT_NE(store, nullptr);

    // Check if the store can be destroyed without issues
    store->destroy(store->self);
}
