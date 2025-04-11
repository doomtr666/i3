#pragma once

#include "native/core/log.h"

#include "vk_backend.h"

#include <vulkan/vulkan.h>

#define VK_VALIDATION_LAYER_NAME "VK_LAYER_KHRONOS_validation"

// logger
i3_logger_i* i3_vk_get_logger();

#define i3_vk_log_fatal(...) do { i3_log_err(i3_vk_get_logger(), __VA_ARGS__); abort(); } while (0)

// vk error string
static const char* i3_vk_result_to_string(VkResult result);

// vk check
void i3_vk_check__(VkResult result, const char* file, int line);
#define i3_vk_check(result) i3_vk_check__(result, __FILE__, __LINE__)


