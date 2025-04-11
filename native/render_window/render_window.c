#include "render_window.h"

#include "native/core/log.h"

#if I3_PLATFORM == I3_PLATFORM_WINDOWS
#define GLFW_EXPOSE_NATIVE_WIN32
#endif

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

typedef enum {
    I3_RENDER_WINDOW_API_NONE = 0,
    I3_RENDER_WINDOW_API_VULKAN,
} i3_render_window_api_t;

// render window interface

struct i3_render_window_o
{
    i3_render_window_i iface;
    i3_render_window_api_t api;
    GLFWwindow* window;

    // vulkan specific
    VkInstance vk_instance;
    VkSurfaceKHR surface;
};

static void i3_render_window_glfw_error_callback(int error_code, const char* description)
{
    i3_logger_i* log = i3_get_logger(I3_RENDER_WINDOW_LOGGER_NAME);

    i3_log_err(log, "GLFW error %d: %s", error_code, description);
}

static void* i3_render_window_get_vk_surface(i3_render_window_o* self)
{
    assert(self != NULL);
    if (self->api != I3_RENDER_WINDOW_API_VULKAN)
        return NULL;
    return self->surface;
}

static void* i3_render_window_get_native_handle(i3_render_window_o* self)
{
    assert(self != NULL);

    return glfwGetWin32Window(self->window);
}

static bool i3_render_window_should_close(i3_render_window_o* self)
{
    assert(self != NULL);

    return glfwWindowShouldClose(self->window);
}

static void i3_render_window_destroy(i3_render_window_o* self)
{
    assert(self != NULL);

    if (self->surface != NULL)
        vkDestroySurfaceKHR(self->vk_instance, self->surface, NULL);

    glfwDestroyWindow(self->window);
    i3_free(self);
}

static i3_render_window_i render_window_iface_ = {
    .self = NULL,
    .get_vk_surface = i3_render_window_get_vk_surface,
    .get_native_handle = i3_render_window_get_native_handle,
    .should_close = i3_render_window_should_close,
    .destroy = i3_render_window_destroy
};

static void i3_render_window_init_glfw()
{
    static bool i3_glfw_initialized = false;

    if (!i3_glfw_initialized)
    {
        if (glfwInit() != GLFW_TRUE)
            return;
        glfwSetErrorCallback(i3_render_window_glfw_error_callback);
        i3_glfw_initialized = true;
    }
}

// render window functions

i3_render_window_i* i3_render_window_create_vulkan(void* vk_instance, const char* title, uint32_t width, uint32_t height)
{
    assert(title != NULL);

    i3_logger_i* log = i3_get_logger(I3_RENDER_WINDOW_LOGGER_NAME);

    // ensure GLFW is initialized
    i3_render_window_init_glfw();

    // check Vulkan support
    if (!glfwVulkanSupported())
    {
        i3_log_err(log, "Vulkan not supported");
        return NULL;
    }

    i3_render_window_o* self = i3_alloc(sizeof(i3_render_window_o));
    assert(self != NULL);
    self->iface = render_window_iface_;
    self->iface.self = self;
    self->api = I3_RENDER_WINDOW_API_VULKAN;
    self->vk_instance = vk_instance;

    // no OpenGL context
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    self->window = glfwCreateWindow(width, height, title, NULL, NULL);
    assert(self->window != NULL);

    // create Vulkan surface
    if (glfwCreateWindowSurface(self->vk_instance, self->window, NULL, &self->surface) != VK_SUCCESS)
    {
        i3_log_err(log, "Failed to create Vulkan surface");
        i3_render_window_destroy(self);
        i3_free(self);
        return NULL;
    }

    return &self->iface;
}

void i3_render_window_poll_events()
{
    i3_render_window_init_glfw();

    glfwPollEvents();
}

const char** i3_render_window_get_required_vk_instance_extensions(uint32_t* count)
{
    assert(count != NULL);

    i3_render_window_init_glfw();

    if (!glfwVulkanSupported())
    {
        i3_logger_i* log = i3_get_logger(I3_RENDER_WINDOW_LOGGER_NAME);
        i3_log_err(log, "Vulkan not supported");
        return NULL;
    }

    return glfwGetRequiredInstanceExtensions(count);
}
