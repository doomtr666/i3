_DEFAULT_NAME = "vcpkg"

def _vcpkg_rule_impl(ctx):

    # download vcpkg
    ctx.download_and_extract("https://github.com/microsoft/vcpkg/archive/refs/tags/{0}.zip".format(ctx.attr.version))

    # create symlinlk to vcpkg_root
    ctx.symlink(
       "vcpkg-{0}".format(ctx.attr.version),
        "vcpkg_root",
    )

    # execution of vcpkg bootstrap
    result = ctx.execute([
        "vcpkg_root/bootstrap-vcpkg.bat",
        "-disableMetrics",
    ])

    if result.return_code != 0:
        fail("unable to bootstrap vcpkg", result.stderr)

    # install glfw3
    result = ctx.execute([
        "vcpkg_root/vcpkg",
        "install",
        "--triplet=x64-windows",
        "gtest",
        "glfw3",
        "shader-slang",
    ])

    if result.return_code != 0:
        fail("unable to install glfw3", result.stderr)

    # generate build file
    ctx.file(
        "BUILD",
        content = """
config_setting(
    name = "dbg_mode",
    values = {
        "compilation_mode": "dbg",
    },
)

cc_library(
    name="gtest",
    hdrs = glob(["vcpkg_root/installed/x64-windows/include/gtest/**/*.h"]),
    strip_include_prefix = "vcpkg_root/installed/x64-windows/include",
    srcs= select({
        ":dbg_mode": [
            "vcpkg_root/installed/x64-windows/debug/lib/manual-link/gtest_main.lib",
            "vcpkg_root/installed/x64-windows/debug/lib/gtest.lib",
            "vcpkg_root/installed/x64-windows/debug/bin/gtest_main.dll",
            "vcpkg_root/installed/x64-windows/debug/bin/gtest.dll",
        ],
        "//conditions:default": [
            "vcpkg_root/installed/x64-windows/lib/manual-link/gtest_main.lib",
            "vcpkg_root/installed/x64-windows/lib/gtest.lib",
            "vcpkg_root/installed/x64-windows/bin/gtest_main.dll",
            "vcpkg_root/installed/x64-windows/bin/gtest.dll",
        ],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name="glfw3",
    hdrs = glob(["vcpkg_root/installed/x64-windows/include/GLFW/*.h"]),
    strip_include_prefix = "vcpkg_root/installed/x64-windows/include",
    srcs= select({
        ":dbg_mode": [
            "vcpkg_root/installed/x64-windows/debug/lib/glfw3dll.lib",
            "vcpkg_root/installed/x64-windows/debug/bin/glfw3.dll",
        ],
        "//conditions:default": [
            "vcpkg_root/installed/x64-windows/lib/glfw3dll.lib",
            "vcpkg_root/installed/x64-windows/bin/glfw3.dll",
        ],
    }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "slangc",
    srcs = ["vcpkg_root/installed/x64-windows/tools/shader-slang/slangc.exe"],
    visibility = ["//visibility:public"],
)

""")

vcpkg_rule = repository_rule(
    local = True,
    attrs = {
        "version": attr.string(mandatory = True),
    },
    implementation = _vcpkg_rule_impl,
)

def _vcpkg_impl(ctx):
    tags = ctx.modules[0].tags.install[0]
    vcpkg_rule(name = tags.name, version = tags.version)

vcpkg = module_extension(
    implementation = _vcpkg_impl,
    tag_classes = {
        "install": tag_class(attrs = {
            "name": attr.string(
                doc = "Base name for generated repositories",
                default = _DEFAULT_NAME,
            ),
            "version": attr.string(
                doc = "Version of vcpkg",
            ),
        }),
    },
)
