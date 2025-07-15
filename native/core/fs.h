#pragma once

#include "common.h"

#define I3_MAX_PATH_LENGTH 4096

typedef enum
{
    I3_DIR_ENTRY_TYPE_UNKNOWN = 0,
    I3_DIR_ENTRY_TYPE_FILE = 1,
    I3_DIR_ENTRY_TYPE_DIR = 2,
} i3_dir_entry_type_t;

typedef struct
{
    i3_dir_entry_type_t type;
    const char* path;
} i3_dir_entry_t;

typedef struct i3_dir_t i3_dir_t;

// file system operations
bool i3_file_exists(const char* file_name);
bool i3_remove_file(const char* file_name);
bool i3_rename_file(const char* from_path, const char* to_path);
bool i3_dir_exists(const char* path);
bool i3_make_dir(const char* path);
bool i3_remove_dir(const char* path);
bool i3_get_cwd(char* buffer, size_t buffer_size);
bool i3_set_cwd(const char* path);
bool i3_get_exe_path(char* buffer, size_t buffer_size);

// directory exploration
i3_dir_t* i3_open_dir(const char* path);
void i3_close_dir(i3_dir_t* dir);
bool i3_dir_next(i3_dir_t* dir, i3_dir_entry_t* entry);
void i3_dir_rewind(i3_dir_t* dir);

// path manipulation
bool i3_get_basename(const char* path, char* buffer, size_t buffer_size);
bool i3_get_dirname(const char* path, char* buffer, size_t buffer_size);
bool i3_join_paths(const char* path1, const char* path2, char* buffer, size_t buffer_size);
bool i3_normalize_path(const char* path, char* buffer, size_t buffer_size);