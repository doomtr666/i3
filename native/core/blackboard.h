#pragma once

#include "arena.h"
#include "hashtable.h"

#define I3_BLACKBOARD_BLOCK_SIZE (16 * I3_KB)
#define I3_BLACKBOARD_ENTRY_MAX_KEY_LENGTH 64

typedef struct i3_blackboard_entry_t
{
    char key[I3_BLACKBOARD_ENTRY_MAX_KEY_LENGTH];
    uint32_t size;
    uint8_t data[];
} i3_blackboard_entry_t;

typedef struct i3_blackboard_t
{
    i3_arena_t entry_store;
    i3_hashtable_t entry_table;
} i3_blackboard_t;

static inline void i3_blackboard_init(i3_blackboard_t* blackboard);
static inline void i3_blackboard_destroy(i3_blackboard_t* blackboard);

static inline bool i3_blackboard_get(i3_blackboard_t* blackboard, const char* key, void* data);
static inline bool i3_blackboard_put(i3_blackboard_t* blackboard, const char* key, void* data, uint32_t size);

// implementation

static inline void i3_blackboard_init(i3_blackboard_t* blackboard)
{
    i3_arena_init(&blackboard->entry_store, I3_BLACKBOARD_BLOCK_SIZE);
    i3_hashtable_init(&blackboard->entry_table);
}

static inline void i3_blackboard_destroy(i3_blackboard_t* blackboard)
{
    assert(blackboard != NULL);
    i3_arena_destroy(&blackboard->entry_store);
    i3_hashtable_destroy(&blackboard->entry_table);
}

static inline bool i3_blackboard_get(i3_blackboard_t* blackboard, const char* key, void* data)
{
    assert(blackboard != NULL);
    assert(key != NULL);
    assert(data != NULL);

    i3_blackboard_entry_t* entry =
        (i3_blackboard_entry_t*)i3_hashtable_find(&blackboard->entry_table, key, strlen(key));
    if (entry == NULL)
        return false;

    memcpy(data, entry->data, entry->size);
    return true;
}

static inline bool i3_blackboard_put(i3_blackboard_t* blackboard, const char* key, void* data, uint32_t size)
{
    assert(blackboard != NULL);
    assert(key != NULL);
    assert(data != NULL);
    assert(size > 0);

    if (size > I3_BLACKBOARD_ENTRY_MAX_KEY_LENGTH - sizeof(i3_blackboard_entry_t))
        return false;  // size too large for a single entry

    i3_blackboard_entry_t* entry =
        (i3_blackboard_entry_t*)i3_arena_alloc(&blackboard->entry_store, sizeof(i3_blackboard_entry_t) + size);
    if (entry == NULL)
        return false;  // allocation failed

    strncpy(entry->key, key, I3_BLACKBOARD_ENTRY_MAX_KEY_LENGTH);
    entry->size = size;
    memcpy(entry->data, data, size);

    return i3_hashtable_insert(&blackboard->entry_table, entry->key, strlen(entry->key), entry);
}