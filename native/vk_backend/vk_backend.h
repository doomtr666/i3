#pragma once

#include "native/core/common.h"
#include "native/render_backend/render_backend.h"

#define I3_VK_BACKEND_LOGGER_NAME "vk_backend"

I3_EXPORT i3_render_backend_i* i3_vk_backend_create(bool enable_validation);
