#pragma once

#include "common.h"

#define I3_VK_BACKEND_INST_EXTS()                                                 \
    /* debug utils */                                                             \
    I3_VK_BACKEND_INST_EXT_NAME(VK_EXT_debug_utils)                               \
    I3_VK_BACKEND_INST_EXT_FN(vkSetDebugUtilsObjectNameEXT)                       \
    I3_VK_BACKEND_INST_EXT_FN(vkSetDebugUtilsObjectTagEXT)                        \
    I3_VK_BACKEND_INST_EXT_FN(vkQueueBeginDebugUtilsLabelEXT)                     \
    I3_VK_BACKEND_INST_EXT_FN(vkQueueEndDebugUtilsLabelEXT)                       \
    I3_VK_BACKEND_INST_EXT_FN(vkQueueInsertDebugUtilsLabelEXT)                    \
    I3_VK_BACKEND_INST_EXT_FN(vkCmdBeginDebugUtilsLabelEXT)                       \
    I3_VK_BACKEND_INST_EXT_FN(vkCmdEndDebugUtilsLabelEXT)                         \
    I3_VK_BACKEND_INST_EXT_FN(vkCmdInsertDebugUtilsLabelEXT)                      \
    I3_VK_BACKEND_INST_EXT_FN(vkCreateDebugUtilsMessengerEXT)                     \
    I3_VK_BACKEND_INST_EXT_FN(vkDestroyDebugUtilsMessengerEXT)                    \
    I3_VK_BACKEND_INST_EXT_FN(vkSubmitDebugUtilsMessageEXT)

typedef struct
{
#define I3_VK_BACKEND_INST_EXT_NAME(ext_name) bool ext_name##_supported;
#define I3_VK_BACKEND_INST_EXT_FN(func_name) PFN_##func_name func_name;

    I3_VK_BACKEND_INST_EXTS()

#undef I3_VK_BACKEND_INST_EXT_NAME
#undef I3_VK_BACKEND_INST_EXT_FN

} i3_vk_backend_instance_ext_t;

void i3_vk_backend_instance_ext_load(VkInstance instance, i3_vk_backend_instance_ext_t* ext);
bool i3_vk_backend_is_instance_ext_supported(const char* name);