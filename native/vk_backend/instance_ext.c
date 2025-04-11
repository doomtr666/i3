#include "instance_ext.h"

void i3_vk_backend_instance_ext_load(VkInstance instance, i3_vk_backend_instance_ext_t* ext)
{
    i3_logger_i* log = i3_vk_get_logger();

    bool* ext_supported;

#define I3_VK_BACKEND_INST_EXT_NAME(ext_name)         \
    ext_supported = &ext->ext_name##_supported; \
    *ext_supported = true;
#define I3_VK_BACKEND_INST_EXT_FN(func_name) \
    *ext_supported &= ((ext->func_name = (PFN_##func_name)vkGetInstanceProcAddr(instance, #func_name)) != NULL);
    I3_VK_BACKEND_INST_EXTS()
#undef I3_VK_BACKEND_INST_EXT_NAME
#undef I3_VK_BACKEND_INST_EXT_FN

#define I3_VK_BACKEND_INST_EXT_NAME(ext_name) \
    i3_log_dbg(log, "Instance extension " #ext_name " loaded: %s", ext->ext_name##_supported ? "TRUE" : "FALSE");
#define I3_VK_BACKEND_INST_EXT_FN(func_name)
        I3_VK_BACKEND_INST_EXTS()
#undef I3_VK_BACKEND_INST_EXT_NAME
#undef I3_VK_BACKEND_INST_EXT_FN
}

bool i3_vk_backend_is_instance_ext_supported(const char* name)
{
#define I3_VK_BACKEND_INST_EXT_NAME(ext_name) \
    if (!strcmp(name, #ext_name))       \
        return true;
#define I3_VK_BACKEND_INST_EXT_FN(func_name)
    I3_VK_BACKEND_INST_EXTS()
#undef I3_VK_BACKEND_INST_EXT_NAME
#undef I3_VK_BACKEND_INST_EXT_FN
        return false;
}