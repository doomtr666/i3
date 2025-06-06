#pragma once

#include "hash.h"

typedef struct i3_hashtable_t i3_hashtable_t;

static inline void i3_hashtable_init(i3_hashtable_t* ht);
static inline void i3_hashtable_destroy(i3_hashtable_t* ht);
static inline void i3_hashtable_clear(i3_hashtable_t* ht);
static inline bool i3_hashtable_insert_hash(i3_hashtable_t* ht,
                                            uint32_t hash,
                                            const void* key,
                                            uint32_t key_size,
                                            void* value);
static inline void* i3_hashtable_find_hash(i3_hashtable_t* ht, uint32_t hash, const void* key, uint32_t key_size);
static inline bool i3_hashtable_remove_hash(i3_hashtable_t* ht, uint32_t hash, const void* key, uint32_t key_size);
static inline bool i3_hashtable_insert(i3_hashtable_t* ht, const void* key, uint32_t key_size, void* value);
static inline void* i3_hashtable_find(i3_hashtable_t* ht, const void* key, uint32_t key_size);
static inline bool i3_hashtable_remove(i3_hashtable_t* ht, const void* key, uint32_t key_size);
static inline uint32_t i3_hashtable_count(i3_hashtable_t* ht);

// implementation

#define I3_HASHTABLE_MIN_CAPACITY 16
#define I3_HASHTABLE_MAX_LOAD_FACTOR 0.7f
#define I3_HASHTABLE_GROW_FACTOR 2

#define I3_HASHTABLE_TOMBSTONE 0xffffffffu

typedef struct i3_hashtable_entry_t
{
    uint32_t hash;
    uint32_t key_size;
    const void* key;
    void* value;
} i3_hashtable_entry_t;

struct i3_hashtable_t
{
    i3_hashtable_entry_t* entries;
    uint32_t capacity;
    uint32_t limit;
    uint32_t count;
};

typedef uint32_t (*i3_hashtable_hash_key)(const void* key);
typedef bool (*i3_hashtable_compare_key)(const void* key1, const void* key2);

static inline uint32_t i3_hashmap_wrap__(uint32_t hash, uint32_t capacity)
{
    return hash & (capacity - 1);
}

static inline void i3_hashtable_grow__(i3_hashtable_t* ht)
{
    assert(ht != NULL);

    // grow the table
    uint32_t new_capacity = ht->capacity * I3_HASHTABLE_GROW_FACTOR;
    if (new_capacity < I3_HASHTABLE_MIN_CAPACITY)
        new_capacity = I3_HASHTABLE_MIN_CAPACITY;

    i3_hashtable_entry_t* new_entries = (i3_hashtable_entry_t*)i3_calloc(new_capacity, sizeof(i3_hashtable_entry_t));
    assert(new_entries != NULL);

    // rehash all entries
    for (uint32_t i = 0; i < ht->capacity; i++)
    {
        i3_hashtable_entry_t* entry = &ht->entries[i];
        if (entry->key != NULL)
        {
            uint32_t index = i3_hashmap_wrap__(entry->hash, new_capacity);
            while (new_entries[index].key != NULL)
                index = i3_hashmap_wrap__(index + 1, new_capacity);

            new_entries[index] = *entry;
        }
    }

    // free old entries
    i3_free(ht->entries);

    ht->capacity = new_capacity;
    ht->limit = (uint32_t)(new_capacity * I3_HASHTABLE_MAX_LOAD_FACTOR);
    ht->entries = new_entries;
}

static inline void i3_hashtable_init(i3_hashtable_t* ht)
{
    assert(ht != NULL);

    ht->entries = NULL;
    ht->capacity = 0;
    ht->limit = 0;
    ht->count = 0;
}

static inline void i3_hashtable_destroy(i3_hashtable_t* ht)
{
    assert(ht != NULL);

    i3_free(ht->entries);
}

static inline void i3_hashtable_clear(i3_hashtable_t* ht)
{
    assert(ht != NULL);

    i3_free(ht->entries);

    ht->entries = NULL;
    ht->capacity = 0;
    ht->limit = 0;
    ht->count = 0;
}

static inline bool i3_hashtable_insert_hash(i3_hashtable_t* ht,
                                            uint32_t hash,
                                            const void* key,
                                            uint32_t key_size,
                                            void* value)
{
    assert(ht != NULL);
    assert(key != NULL);
    assert(key_size > 0);

    if (ht->count >= ht->limit)
        i3_hashtable_grow__(ht);

    // insert the new entry
    uint32_t index = i3_hashmap_wrap__(hash, ht->capacity);

    // find an empty slot or the existing key
    for (;;)
    {
        i3_hashtable_entry_t* entry = &ht->entries[index];

        if (entry->hash == hash && entry->key_size == key_size && memcmp(entry->key, key, key_size) == 0)
        {
            entry->value = value;
            return false;
        }

        if (ht->entries[index].key == NULL)
        {
            entry->hash = hash;
            entry->key_size = key_size;
            entry->key = key;
            entry->value = value;
            ht->count++;

            return true;
        }

        index = i3_hashmap_wrap__(index + 1, ht->capacity);
    }
}

static inline void* i3_hashtable_find_hash(i3_hashtable_t* ht, uint32_t hash, const void* key, uint32_t key_size)
{
    assert(ht != NULL);
    assert(key != NULL);
    assert(key_size > 0);

    if (ht->entries == NULL)
        return NULL;

    uint32_t index = i3_hashmap_wrap__(hash, ht->capacity);

    for (;;)
    {
        i3_hashtable_entry_t* entry = &ht->entries[index];

        if (entry->hash == hash && entry->key_size == key_size && memcmp(entry->key, key, key_size) == 0)
            return entry->value;

        if (entry->key == NULL && entry->hash != I3_HASHTABLE_TOMBSTONE)
            return NULL;

        index = i3_hashmap_wrap__(index + 1, ht->capacity);
    }
}

static inline bool i3_hashtable_remove_hash(i3_hashtable_t* ht, uint32_t hash, const void* key, uint32_t key_size)
{
    assert(ht != NULL);
    assert(key != NULL);

    uint32_t index = i3_hashmap_wrap__(hash, ht->capacity);

    for (;;)
    {
        if (ht->entries[index].hash == hash && ht->entries[index].key_size == key_size &&
            memcmp(ht->entries[index].key, key, key_size) == 0)
        {
            ht->entries[index].key = NULL;
            ht->entries[index].value = NULL;
            ht->entries[index].hash = I3_HASHTABLE_TOMBSTONE;
            ht->count--;
            return true;
        }

        if (ht->entries[index].key == NULL && ht->entries[index].hash != I3_HASHTABLE_TOMBSTONE)
            return false;

        index = i3_hashmap_wrap__(index + 1, ht->capacity);
    }

    return false;
}

static inline bool i3_hashtable_insert(i3_hashtable_t* ht, const void* key, uint32_t key_size, void* value)
{
    assert(ht != NULL);
    assert(key != NULL);
    assert(key_size > 0);

    uint32_t hash = i3_hash32(key, key_size, 0);
    return i3_hashtable_insert_hash(ht, hash, key, key_size, value);
}

static inline void* i3_hashtable_find(i3_hashtable_t* ht, const void* key, uint32_t key_size)
{
    assert(ht != NULL);
    assert(key != NULL);
    assert(key_size > 0);

    uint32_t hash = i3_hash32(key, key_size, 0);
    return i3_hashtable_find_hash(ht, hash, key, key_size);
}

static inline bool i3_hashtable_remove(i3_hashtable_t* ht, const void* key, uint32_t key_size)
{
    assert(ht != NULL);
    assert(key != NULL);

    uint32_t hash = i3_hash32(key, key_size, 0);
    return i3_hashtable_remove_hash(ht, hash, key, key_size);
}

static inline uint32_t i3_hashtable_count(i3_hashtable_t* ht)
{
    assert(ht != NULL);

    return ht->count;
}
