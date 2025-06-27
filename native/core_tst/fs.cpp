#include "gtest/gtest.h"

extern "C"
{
#include "native/core/fs.h"
}

TEST(fs, get_cwd)
{
    char cwd_buf[4096];
    EXPECT_TRUE(i3_get_cwd(cwd_buf, sizeof(cwd_buf)));
    EXPECT_GT(strlen(cwd_buf), 0);
}

TEST(fs, set_cwd)
{
    // Get the current working directory
    char cwd_buf[4096];
    EXPECT_TRUE(i3_get_cwd(cwd_buf, sizeof(cwd_buf)));

    // Set the current working directory to the same path
    EXPECT_TRUE(i3_set_cwd(cwd_buf));

    // Verify that the current working directory is still the same
    char new_cwd_buf[4096];
    EXPECT_TRUE(i3_get_cwd(new_cwd_buf, sizeof(new_cwd_buf)));
    EXPECT_STREQ(cwd_buf, new_cwd_buf);
}

TEST(fs, basename)
{
    const char* path = "C:\\path\\to\\file.txt";
    char basename_buf[256];
    EXPECT_TRUE(i3_get_basename(path, basename_buf, sizeof(basename_buf)));
    EXPECT_STREQ(basename_buf, "file.txt");

    const char* path2 = "/path/to/file.txt";
    char basename_buf2[256];
    EXPECT_TRUE(i3_get_basename(path2, basename_buf2, sizeof(basename_buf2)));
    EXPECT_STREQ(basename_buf2, "file.txt");

    const char* path3 = "file.txt";
    char basename_buf3[256];
    EXPECT_TRUE(i3_get_basename(path3, basename_buf3, sizeof(basename_buf3)));
    EXPECT_STREQ(basename_buf3, "file.txt");

    const char* path4 = "C:\\path\\to\\";
    char basename_buf4[256];
    EXPECT_TRUE(i3_get_basename(path4, basename_buf4, sizeof(basename_buf4)));
    EXPECT_STREQ(basename_buf4, "");

    const char* path5 = "";
    char basename_buf5[256];
    EXPECT_TRUE(i3_get_basename(path5, basename_buf5, sizeof(basename_buf5)));
    EXPECT_STREQ(basename_buf5, "");
}

TEST(fs, dirname)
{
    const char* path = "C:\\path\\to\\file.txt";
    char dirname_buf[256];
    EXPECT_TRUE(i3_get_dirname(path, dirname_buf, sizeof(dirname_buf)));
    EXPECT_STREQ(dirname_buf, "C:\\path\\to");

    const char* path2 = "/path/to/file.txt";
    char dirname_buf2[256];
    EXPECT_TRUE(i3_get_dirname(path2, dirname_buf2, sizeof(dirname_buf2)));
    EXPECT_STREQ(dirname_buf2, "/path/to");

    const char* path3 = "file.txt";
    char dirname_buf3[256];
    EXPECT_TRUE(i3_get_dirname(path3, dirname_buf3, sizeof(dirname_buf3)));
    EXPECT_STREQ(dirname_buf3, "");

    const char* path4 = "C:\\path\\to\\";
    char dirname_buf4[256];
    EXPECT_TRUE(i3_get_dirname(path4, dirname_buf4, sizeof(dirname_buf4)));
    EXPECT_STREQ(dirname_buf4, "C:\\path\\to");

    const char* path5 = "";
    char dirname_buf5[256];
    EXPECT_TRUE(i3_get_dirname(path5, dirname_buf5, sizeof(dirname_buf5)));
    EXPECT_STREQ(dirname_buf5, "");

    const char* path6 =
        "C:\\users\\chris\\_bazel_chris\\e7dcpzc7\\execroot\\_main\\bazel-out\\x64_windows-dbg\\bin\\samples\\game_"
        "draw_cube\\game_draw_cube.exe";
    char dirname_buf6[256];
    EXPECT_TRUE(i3_get_dirname(path6, dirname_buf6, sizeof(dirname_buf6)));
    EXPECT_STREQ(dirname_buf6,
                 "C:\\users\\chris\\_bazel_chris\\e7dcpzc7\\execroot\\_main\\bazel-out\\x64_windows-"
                 "dbg\\bin\\samples\\game_draw_cube");
}

TEST(fs, join_paths)
{
    const char* path1 = "C:\\path\\to";
    const char* path2 = "file.txt";
    char joined_buf[256];
    EXPECT_TRUE(i3_join_paths(path1, path2, joined_buf, sizeof(joined_buf)));
    EXPECT_STREQ(joined_buf, "C:/path/to/file.txt");

    const char* path3 = "/path/to";
    const char* path4 = "file.txt";
    char joined_buf2[256];
    EXPECT_TRUE(i3_join_paths(path3, path4, joined_buf2, sizeof(joined_buf2)));
    EXPECT_STREQ(joined_buf2, "/path/to/file.txt");

    const char* path5 = "file.txt";
    const char* path6 = "";
    char joined_buf3[256];
    EXPECT_TRUE(i3_join_paths(path5, path6, joined_buf3, sizeof(joined_buf3)));
    EXPECT_STREQ(joined_buf3, "file.txt");
}