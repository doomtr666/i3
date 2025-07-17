_DEFAULT_NAME = "vulkan_sdk"

def _vulkan_sdk_rule_impl(ctx):
    sdk_path = ctx.getenv("VULKAN_SDK", "")
    if not sdk_path:
        fail("VULKAN_SDK environment variable is not set. Please install the latest Vulkan SDK.")

    ctx.symlink(sdk_path, "vulkan_sdk")

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
    },
    implementation = _vulkan_sdk_rule_impl,
)

def _vulkan_sdk_impl(ctx):
    tags = ctx.modules[0].tags.install[0]
    vulkan_sdk_rule(name = tags.name)

vulkan_sdk = module_extension(
    implementation = _vulkan_sdk_impl,
    tag_classes = {
        "install": tag_class(attrs = {
            "name": attr.string(
                doc = "Base name for generated repositories",
                default = _DEFAULT_NAME,
            ),
        }),
    },
)
