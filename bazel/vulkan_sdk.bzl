load("//bazel:seven_zip.bzl", "extract_7zip")

_DEFAULT_NAME = "vulkan_sdk"

def _vulkan_sdk_rule_impl(ctx):

    # download vulkan_sdk
    ctx.download(
        url = "https://sdk.lunarg.com/sdk/download/{0}/windows/VulkanSDK-{0}-Installer.exe".format(ctx.attr.version),
        output = "vulkan_sdk.exe",
    )

    # extract vulkan_sdk self extracting archive
    extract_7zip(
        ctx = ctx,
        src = "vulkan_sdk.exe",
        dst = "vulkan_sdk",
    )

    # create BUILD file
    ctx.file("BUILD", """

cc_library(
    name = "core",
    strip_include_prefix= "vulkan_sdk/include/",
    hdrs = glob(["vulkan_sdk/include/vulkan/*.h", "vulkan_sdk/include/vk_video/*.h"]),
    srcs = ["vulkan_sdk/Lib/vulkan-1.lib"],
    visibility = ["//visibility:public"],
)

""")

vulkan_sdk_rule = repository_rule(
    local = True,
    attrs = {
        "version": attr.string(mandatory = True),
    },
    implementation = _vulkan_sdk_rule_impl,
)

def _vulkan_sdk_impl(ctx):
    tags = ctx.modules[0].tags.install[0]
    vulkan_sdk_rule(name = tags.name, version = tags.version)

vulkan_sdk = module_extension(
    implementation = _vulkan_sdk_impl,
    tag_classes = {
        "install": tag_class(attrs = {
            "name": attr.string(
                doc = "Base name for generated repositories",
                default = _DEFAULT_NAME,
            ),
            "version": attr.string(
                doc = "Version of the VulkanSDK",
            ),
        }),
    },
)
