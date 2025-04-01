_DEFAULT_NAME = "vulkan_sdk"

def _vulkan_sdk_impl(ctx):
    raise("error !!")
    #ctx.download_and_extract("https://sdk.lunarg.com/sdk/download/1.4.309.0/windows/VulkanSDK-1.4.309.0-Installer.exe", "vulkan_sdk")

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