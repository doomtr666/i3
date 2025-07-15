#include "content_store.h"
#include "native/core/arena.h"
#include "native/core/common.h"
#include "native/core/fs.h"
#include "native/core/log.h"

#include <stdio.h>

// content implementation
struct i3_content_o
{
    i3_content_i iface;
    uint32_t size;       // size of the content data
    uint32_t use_count;  // reference count for the content
    uint8_t data[];      // pointer to the content data
};

static const void* i3_content_get_data(i3_content_o* self)
{
    assert(self != NULL);
    return self->data;
}

static uint32_t i3_content_get_size(i3_content_o* self)
{
    assert(self != NULL);
    return self->size;
}

static void i3_content_add_ref(i3_content_o* self)
{
    assert(self != NULL);
    self->use_count++;
}

static void i3_content_release(i3_content_o* self)
{
    assert(self != NULL);

    self->use_count--;
    if (self->use_count > 0)
        return;  // still in use, do not free
    i3_free(self);
}

static i3_content_i i3_content_iface_ = {
    .get_data = i3_content_get_data,
    .get_size = i3_content_get_size,
    .add_ref = i3_content_add_ref,
    .release = i3_content_release,
};

static i3_content_o* i3_create_content(uint32_t size)
{
    // allocate memory for the content object
    i3_content_o* content = (i3_content_o*)i3_alloc(sizeof(i3_content_o) + size);
    assert(content != NULL);

    content->iface = i3_content_iface_;
    content->iface.self = content;
    content->use_count = 1;  // initial reference count
    content->size = size;

    return content;
}

// content store implementation

struct i3_content_store_o
{
    i3_content_store_i iface;
    i3_logger_i* log;  // logger for the content store
};

static i3_content_i* i3_content_store_load(i3_content_store_o* self, const char* path)
{
    assert(self != NULL);
    assert(path != NULL);

    // open the file
    FILE* file = fopen(path, "rb");
    if (!file)
    {
        i3_log_err(self->log, "Failed to open file: %s", path);
        return NULL;
    }

    // get the file size
    fseek(file, 0, SEEK_END);
    uint32_t file_size = (uint32_t)ftell(file);
    fseek(file, 0, SEEK_SET);

    if (file_size <= 0)
    {
        i3_log_err(self->log, "File is empty or invalid: %s", path);
        fclose(file);
        return NULL;
    }

    // allocate memory for the content
    i3_content_o* content = i3_create_content(file_size);

    // read the file content
    size_t bytes_read = fread(content->data, 1, file_size, file);
    fclose(file);

    i3_log_dbg(self->log, "Loaded content from file: %s, size: %u bytes", path, content->size);

    return &content->iface;
}

static void i3_content_store_destroy(i3_content_store_o* self)
{
    assert(self != NULL);

    // free the content store object
    i3_free(self);
}

static i3_content_store_o i3_content_store_iface_ = {
    .iface = {
        .load = i3_content_store_load, 
        .destroy = i3_content_store_destroy,
    },

};

static bool i3_search_for_executable_in_runfiles(i3_arena_t* arena,
                                                 i3_logger_i* log,
                                                 const char* cwd,
                                                 const char* exe_name,
                                                 char* found_path,
                                                 size_t found_path_size)
{
    assert(cwd != NULL);
    assert(exe_name != NULL);
    assert(found_path != NULL);
    assert(found_path_size > 0);

    i3_dir_t* dir = i3_open_dir(cwd);
    if (!dir)
    {
        i3_log_err(log, "Failed to open directory: %s", cwd);
        return false;
    }

    i3_dir_entry_t entry;
    while (i3_dir_next(dir, &entry))
    {
        if (!strcmp(entry.path, ".") || !strcmp(entry.path, ".."))
            continue;

        if (entry.type == I3_DIR_ENTRY_TYPE_FILE && !strcmp(entry.path, exe_name))
        {
            if (!i3_normalize_path(cwd, found_path, found_path_size))
            {
                i3_log_err(log, "Failed to normalize path: %s", cwd);
                exit(EXIT_FAILURE);
            }

            i3_close_dir(dir);
            return true;
        }
        if (entry.type == I3_DIR_ENTRY_TYPE_DIR)
        {
            char* subdir_path = i3_arena_alloc(arena, I3_MAX_PATH_LENGTH);
            if (!i3_join_paths(cwd, entry.path, subdir_path, I3_MAX_PATH_LENGTH))
            {
                i3_log_err(log, "Failed to join paths: %s and %s", cwd, entry.path);
                exit(EXIT_FAILURE);
            }

            // Recursively search in subdirectories
            if (i3_search_for_executable_in_runfiles(arena, log, subdir_path, exe_name, found_path, found_path_size))
            {
                i3_close_dir(dir);
                return true;
            }
        }
    }

    i3_close_dir(dir);
    return false;
}

i3_content_store_i* i3_content_store_create()
{
    i3_content_store_o* store = (i3_content_store_o*)i3_alloc(sizeof(i3_content_store_o));
    *store = i3_content_store_iface_;
    store->iface.self = store;

    // initlialize the logger
    store->log = i3_get_logger(I3_CONTENT_STORE_LOGGER_NAME);

    i3_arena_t arena;
    i3_arena_init(&arena, 1024 * 1024);  // Initialize arena with 1MB block size

    // const char* exe_path = argv[0];
    char* exe_path = i3_arena_alloc(&arena, I3_MAX_PATH_LENGTH);
    if (!i3_get_exe_path(exe_path, I3_MAX_PATH_LENGTH))
    {
        i3_log_err(store->log, "Failed to get executable path");
        exit(EXIT_FAILURE);
    }

    char* cwd = i3_arena_alloc(&arena, I3_MAX_PATH_LENGTH);

    if (!i3_get_cwd(cwd, I3_MAX_PATH_LENGTH))
    {
        i3_log_err(store->log, "Failed to get current working directory");
        exit(EXIT_FAILURE);
    }

    char* exe_name = i3_arena_alloc(&arena, I3_MAX_PATH_LENGTH);
    if (!i3_get_basename(exe_path, exe_name, I3_MAX_PATH_LENGTH))
    {
        i3_log_err(store->log, "Failed to get executable name from path: %s", exe_path);
        exit(EXIT_FAILURE);
    }

    char* exe_dir = i3_arena_alloc(&arena, I3_MAX_PATH_LENGTH);
    if (!i3_get_dirname(exe_path, exe_dir, I3_MAX_PATH_LENGTH))
    {
        i3_log_err(store->log, "Failed to get executable directory from path: %s", exe_path);
        exit(EXIT_FAILURE);
    }

    // log the paths
    i3_log_dbg(store->log, "Executable name: %s", exe_name);
    i3_log_dbg(store->log, "Executable path: %s", exe_path);
    i3_log_dbg(store->log, "Executable directory: %s", exe_dir);
    i3_log_dbg(store->log, "Current working directory: %s", cwd);

    if (strcmp(exe_dir, cwd))
    {
        i3_log_wrn(store->log,
                   "Executable directory does not match current working directory. Probably running with bazel.");

        // Attempt to find the executable in the runfiles
        char* runfiles_exe_dir = i3_arena_alloc(&arena, I3_MAX_PATH_LENGTH);

        if (i3_search_for_executable_in_runfiles(&arena, store->log, cwd, exe_name, runfiles_exe_dir,
                                                 I3_MAX_PATH_LENGTH))
        {
            i3_log_dbg(store->log, "Setting cwd to: %s", runfiles_exe_dir);
            // Change working directory to the directory containing the executable
            if (!i3_set_cwd(runfiles_exe_dir))
            {
                i3_log_err(store->log, "Failed to change working directory to: %s", runfiles_exe_dir);
                exit(EXIT_FAILURE);
            }
        }
        else
        {
            i3_log_err(store->log, "Executable not found in runfiles. Cannot change working directory.");
            exit(EXIT_FAILURE);
        }
    }

    i3_arena_destroy(&arena);

    i3_log_inf(store->log, "Content store created");

    return &store->iface;
}