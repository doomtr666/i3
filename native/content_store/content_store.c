#include "content_store.h"
#include "native/core/arena.h"
#include "native/core/common.h"
#include "native/core/fs.h"
#include "native/core/log.h"

// content implementation
struct i3_content_o
{
    i3_content_i iface;
    uint32_t size;   // size of the content data
    uint8_t data[];  // pointer to the content data
};

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

    return NULL;  // This function should load content from the specified path.
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

i3_content_store_i* i3_content_store_create(int argc, char** argv)
{
    i3_content_store_o* store = (i3_content_store_o*)i3_alloc(sizeof(i3_content_store_o));
    *store = i3_content_store_iface_;
    store->iface.self = store;

    // initlialize the logger
    store->log = i3_get_logger(I3_CONTENT_STORE_LOGGER_NAME);

    i3_arena_t arena;
    i3_arena_init(&arena, 1024 * 1024);  // Initialize arena with 1MB block size

    const char* exe_path = argv[0];
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