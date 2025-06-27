#include < windows.h>
#include <direct.h>

#include "fs.h"

bool i3_file_exists(const char* file_name)
{
    assert(file_name != NULL);

    DWORD dwAttrib = GetFileAttributesA(file_name);
    return (dwAttrib != INVALID_FILE_ATTRIBUTES && !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

bool i3_remove_file(const char* file_name)
{
    assert(file_name != NULL);

    return DeleteFileA(file_name) == TRUE;
}

bool i3_rename_file(const char* from_path, const char* to_path)
{
    assert(from_path != NULL);
    assert(to_path != NULL);

    return MoveFileExA(from_path, to_path, MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED) == TRUE;
}

bool i3_dir_exists(const char* path)
{
    assert(path != NULL);

    DWORD dwAttrib = GetFileAttributesA(path);
    return (dwAttrib != INVALID_FILE_ATTRIBUTES && (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

bool i3_make_dir(const char* path)
{
    assert(path != NULL);

    return CreateDirectoryA(path, NULL) == TRUE;
}

bool i3_remove_dir(const char* path)
{
    assert(path != NULL);

    return RemoveDirectoryA(path) == TRUE;
}

struct i3_dir_t
{
    char* search_path;
    HANDLE handle;
    WIN32_FIND_DATA find_data;
};

i3_dir_t* i3_open_dir(const char* path)
{
    assert(path != NULL);

    // check path is a directory
    if (!i3_dir_exists(path))
        return NULL;

    // allocate dir structure
    i3_dir_t* dir = i3_alloc(sizeof(i3_dir_t));
    assert(dir != NULL);  // Ensure memory allocation was successful

    // initialize dir structure
    dir->search_path = NULL;
    dir->handle = INVALID_HANDLE_VALUE;

    // create search path, append wildcard to the path
    size_t path_len = strlen(path);
    dir->search_path = i3_alloc(path_len + 3);
    memcpy(dir->search_path, path, path_len);
    dir->search_path[path_len] = '/';
    dir->search_path[path_len + 1] = '*';
    dir->search_path[path_len + 2] = 0;

    return dir;
}

void i3_close_dir(i3_dir_t* dir)
{
    assert(dir != NULL);

    if (dir->handle != INVALID_HANDLE_VALUE)
        FindClose(dir->handle);
    i3_free(dir->search_path);

    i3_free(dir);
}

bool i3_dir_next(i3_dir_t* dir, i3_dir_entry_t* entry)
{
    assert(dir != NULL);
    assert(entry != NULL);

    // iterate through file
    if (dir->handle == INVALID_HANDLE_VALUE)
    {
        dir->handle = FindFirstFileA(dir->search_path, &dir->find_data);
        if (dir->handle == INVALID_HANDLE_VALUE)
            return false;
    }
    else
    {
        if (!FindNextFileA(dir->handle, &dir->find_data))
            return false;
    }

    entry->path = (const char*)dir->find_data.cFileName;
    if (dir->find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
        entry->type = I3_DIR_ENTRY_TYPE_DIR;
    else if (dir->find_data.dwFileAttributes & FILE_ATTRIBUTE_NORMAL
             || dir->find_data.dwFileAttributes & FILE_ATTRIBUTE_ARCHIVE)
        entry->type = I3_DIR_ENTRY_TYPE_FILE;
    else
        entry->type = I3_DIR_ENTRY_TYPE_UNKNOWN;

    return true;
}

void i3_dir_rewind(i3_dir_t* dir)
{
    assert(dir != NULL);

    if (dir->handle != INVALID_HANDLE_VALUE)
    {
        FindClose(dir->handle);
        dir->handle = INVALID_HANDLE_VALUE;
    }
}

bool i3_get_cwd(char* buffer, size_t buffer_size)
{
    assert(buffer != NULL);
    assert(buffer_size > 0);

    if (_getcwd(buffer, buffer_size) == NULL)
        return false;
    return true;
}

bool i3_set_cwd(const char* path)
{
    if (!path)
        return false;  // Invalid path

    if (_chdir((const char*)path) != 0)
        return false;  // Error setting current directory

    return true;
}

static const char* i3_path_separators_ = "\\/";

static inline const char* i3_find_last_sep_(const char* path, uint32_t len)
{
    assert(path != NULL);

    for (const char* p = path + len - 1; p >= path; --p)
    {
        if (*p == '/' || *p == '\\')
            return p;  // Return the last separator found
    }

    return NULL;  // No separator found
}

bool i3_get_basename(const char* path, char* buffer, size_t buffer_size)
{
    assert(path != NULL);
    assert(buffer != NULL);
    assert(buffer_size > 0);

    uint32_t len = strlen(path);

    const char* last_slash = i3_find_last_sep_(path, len);
    if (last_slash == NULL)
    {
        // No separator found, return the whole path as basename
        if (len >= buffer_size)
            return false;  // Buffer too small

        strcpy(buffer, path);
        return true;
    }

    // if the last separator is the last character, return an empty basename
    if (last_slash == path + len - 1)
    {
        buffer[0] = '\0';  // Set buffer to empty string
        return true;
    }

    last_slash++;
    uint32_t basename_len = len - (last_slash - path) - 1;  // Length of the basename

    if (basename_len >= buffer_size)
        return false;  // Buffer too small

    strcpy(buffer, last_slash);
    return true;
}

bool i3_get_dirname(const char* path, char* buffer, size_t buffer_size)
{
    assert(path != NULL);
    assert(buffer != NULL);
    assert(buffer_size > 0);

    uint32_t len = strlen(path);

    const char* last_slash = i3_find_last_sep_(path, len);
    if (last_slash == NULL)
    {
        // No separator found, return empty dirname
        buffer[0] = '\0';
        return true;
    }

    uint32_t dirname_len = last_slash - path;  // Length of the dirname

    if (dirname_len + 1 >= buffer_size)
        return false;  // Buffer too small

    strncpy(buffer, path, dirname_len);
    buffer[dirname_len] = '\0';  // Null-terminate the dirname
    return true;
}

bool i3_join_paths(const char* path1, const char* path2, char* buffer, size_t buffer_size)
{
    assert(path1 != NULL);
    assert(path2 != NULL);
    assert(buffer != NULL);
    assert(buffer_size > 0);

    char tmp_buffer[I3_MAX_PATH_LENGTH];

    size_t len1 = strlen(path1);
    size_t len2 = strlen(path2);
    size_t total_len = len1 + len2 + 2;  // +2 for the separator and null terminator
    if (total_len > sizeof(tmp_buffer))
        return false;  // Buffer too small

    // Copy the first path
    strncpy(tmp_buffer, path1, len1);
    tmp_buffer[len1] = '/';  // Add separator

    // Copy the second path
    strncpy(tmp_buffer + len1 + 1, path2, len2);
    tmp_buffer[len1 + len2 + 1] = '\0';  // Null-terminate the result

    if (i3_normalize_path(tmp_buffer, buffer, buffer_size) == false)
        return false;  // Normalization failed
    return true;
}

bool i3_normalize_path(const char* path, char* buffer, size_t buffer_size)
{
    assert(path != NULL);
    assert(buffer != NULL);
    assert(buffer_size > 0);
    size_t len = strlen(path);
    if (len >= buffer_size)
        return false;  // Buffer too small

    // Copy the path to the buffer
    strcpy(buffer, path);

    // Replace backslashes with forward slashes
    for (size_t i = 0; i < len; i++)
    {
        if (buffer[i] == '\\')
            buffer[i] = '/';
    }

    // remove trailing slashes
    while (len > 0 && (buffer[len - 1] == '/' || buffer[len - 1] == '\\'))
    {
        buffer[len - 1] = '\0';
        len--;
    }

    return true;
}