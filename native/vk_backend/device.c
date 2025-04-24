
#include "native/core/array.h"

#include "buffer.h"
#include "cmd_buffer.h"
#include "cmd_list.h"
#include "descriptor_set_layout.h"
#include "framebuffer.h"
#include "image.h"
#include "image_view.h"
#include "pipeline.h"
#include "pipeline_layout.h"
#include "sampler.h"
#include "shader_module.h"
#include "submission.h"
#include "swapchain.h"
#include "use_list.h"

static void i3_vk_device_destroy(i3_rbk_device_o* self)
{
    i3_vk_device_o* device = (i3_vk_device_o*)self;

    // wait  for  pending submissions
    while (i3_array_count(&device->submissions) > 0)
    {
        i3_vk_device_end_frame(self);
        vkDeviceWaitIdle(device->handle);
    }

    // destroy submissions array
    i3_array_free(&device->submissions);

    // destroy resource pools
    i3_memory_pool_destroy(&device->use_list_block_pool);
    i3_memory_pool_destroy(&device->cmd_list_block_pool);
    i3_memory_pool_destroy(&device->sampler_pool);
    i3_memory_pool_destroy(&device->buffer_pool);
    i3_memory_pool_destroy(&device->image_pool);
    i3_memory_pool_destroy(&device->image_view_pool);
    i3_memory_pool_destroy(&device->descriptor_set_layout_pool);
    i3_memory_pool_destroy(&device->pipeline_layout_pool);
    i3_memory_pool_destroy(&device->framebuffer_pool);
    i3_memory_pool_destroy(&device->shader_module_pool);
    i3_memory_pool_destroy(&device->pipeline_pool);
    i3_memory_pool_destroy(&device->cmd_buffer_pool);
    i3_memory_pool_destroy(&device->submission_pool);

    // destroy command pool
    vkDestroyCommandPool(device->handle, device->cmd_pool, NULL);

    // destroy vma
    vmaDestroyAllocator(device->vma);

    // destroy device
    vkDestroyDevice(device->handle, NULL);

    i3_log_inf(device->log, "Vulkan device destroyed");
    i3_free(device);
}

static i3_vk_device_o i3_vk_device_iface_
    = {.iface = {.create_sampler = i3_vk_device_create_sampler,
                 .create_buffer = i3_vk_device_create_buffer,
                 .create_image = i3_vk_device_create_image,
                 .create_image_view = i3_vk_device_create_image_view,
                 .create_descriptor_set_layout = i3_vk_device_create_descriptor_set_layout,
                 .create_pipeline_layout = i3_vk_device_create_pipeline_layout,
                 .create_framebuffer = i3_vk_device_create_framebuffer,
                 .create_shader_module = i3_vk_device_create_shader_module,
                 .create_graphics_pipeline = i3_vk_device_create_graphics_pipeline,
                 .create_compute_pipeline = i3_vk_device_create_compute_pipeline,
                 .create_swapchain = i3_vk_device_create_swapchain,
                 .create_cmd_buffer = i3_vk_device_create_cmd_buffer,
                 .submit_cmd_buffers = i3_vk_device_submit_cmd_buffers,
                 .present = i3_vk_device_present,
                 .end_frame = i3_vk_device_end_frame,
                 .destroy = i3_vk_device_destroy}};

i3_rbk_device_i* i3_vk_device_create(i3_vk_backend_o* backend, i3_vk_device_desc* device_desc)
{
    assert(backend != NULL);
    assert(device_desc != NULL);

    i3_vk_device_o* device = i3_alloc(sizeof(i3_vk_device_o));
    assert(device != NULL);
    *device = i3_vk_device_iface_;
    device->iface.self = (i3_rbk_device_o*)device;
    device->log = i3_vk_get_logger();
    device->backend = backend;
    device->desc = *device_desc;

    // get queue family properties
    uint32_t qfam_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device_desc->physical_device, &qfam_count, NULL);
    VkQueueFamilyProperties* qfam_props = i3_alloc(qfam_count * sizeof(VkQueueFamilyProperties));
    assert(qfam_props != NULL);
    vkGetPhysicalDeviceQueueFamilyProperties(device_desc->physical_device, &qfam_count, qfam_props);

    // create queues
    i3_array_t queues_ci;
    i3_array_init(&queues_ci, sizeof(VkDeviceQueueCreateInfo));
    float queue_priority = 1;

    uint32_t graphics_qfam = UINT32_MAX;
    uint32_t compute_qfam = UINT32_MAX;
    uint32_t async_compute_qfam = UINT32_MAX;
    uint32_t transfer_qfam = UINT32_MAX;

    for (uint32_t i = 0; i < qfam_count; i++)
    {
        char graphics_flag = (qfam_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0 ? 'G' : '_';
        char compute_flag = (qfam_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 ? 'C' : '_';
        char transfer_flag = (qfam_props[i].queueFlags & VK_QUEUE_TRANSFER_BIT) != 0 ? 'T' : '_';
        char sparse_flag = (qfam_props[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) != 0 ? 'S' : '_';
        char protected_flag = (qfam_props[i].queueFlags & VK_QUEUE_PROTECTED_BIT) != 0 ? 'P' : '_';
        char video_decode_flag = (qfam_props[i].queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) != 0 ? 'D' : '_';
        char video_encode_flag = (qfam_props[i].queueFlags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR) != 0 ? 'E' : '_';
        char optical_flow_flag = (qfam_props[i].queueFlags & VK_QUEUE_OPTICAL_FLOW_BIT_NV) != 0 ? 'F' : '_';

        VkDeviceQueueCreateInfo* queue_ci = i3_array_addn(&queues_ci, 1);
        *queue_ci = (VkDeviceQueueCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = i,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority,
        };

        i3_log_dbg(device->log, "Device queue family %d: %d queue(s), flags 0x%x (%c.%c.%c.%c.%c.%c.%c.%c)", i,
                   qfam_props[i].queueCount, qfam_props[i].queueFlags, graphics_flag, compute_flag, transfer_flag,
                   sparse_flag, protected_flag, video_decode_flag, video_encode_flag, optical_flow_flag);
    }

    // enumarate extensions
    uint32_t ext_count = 0;
    i3_vk_check(vkEnumerateDeviceExtensionProperties(device_desc->physical_device, NULL, &ext_count, NULL));
    VkExtensionProperties* ext_props = i3_alloc(ext_count * sizeof(VkExtensionProperties));
    assert(ext_props != NULL);
    i3_vk_check(vkEnumerateDeviceExtensionProperties(device_desc->physical_device, NULL, &ext_count, ext_props));

    // enabled extensions
    i3_array_t enabled_exts;
    i3_array_init(&enabled_exts, sizeof(const char*));
    for (uint32_t i = 0; i < ext_count; i++)
    {
        const char* ext_name = ext_props[i].extensionName;
        if (i3_vk_device_ext_supported(ext_name))
            i3_array_push(&enabled_exts, &ext_name);
        i3_log_dbg(device->log, "Device extension: %s", ext_name);
    }

    // log enabled extensions
    for (uint32_t i = 0; i < i3_array_count(&enabled_exts); i++)
    {
        const char* ext_name = *(const char**)i3_array_at(&enabled_exts, i);
        i3_log_dbg(device->log, "Enabled device extension: %s", ext_name);
    }

    // create device
    VkDeviceCreateInfo device_ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = i3_array_count(&queues_ci),
        .pQueueCreateInfos = i3_array_data(&queues_ci),
        .enabledExtensionCount = i3_array_count(&enabled_exts),
        .ppEnabledExtensionNames = i3_array_data(&enabled_exts),
    };

    i3_vk_check(vkCreateDevice(device_desc->physical_device, &device_ci, NULL, &device->handle));

    // cleanup
    i3_array_free(&queues_ci);
    i3_array_free(&enabled_exts);
    i3_free(qfam_props);
    i3_free(ext_props);

    // load extensions
    i3_vk_device_ext_load(device->handle, &device->ext);

    // create command pool
    VkCommandPoolCreateInfo cmd_pool_ci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = 0,  // TODO: handle queues
        .flags = 0,
    };
    i3_vk_check(vkCreateCommandPool(device->handle, &cmd_pool_ci, NULL, &device->cmd_pool));

    // get graphics queue
    // TODO: handle queues
    vkGetDeviceQueue(device->handle, 0, 0, &device->graphics_queue);

    // create VMA
    VmaAllocatorCreateInfo allocator_ci = {.physicalDevice = device_desc->physical_device,
                                           .device = device->handle,
                                           .instance = backend->instance,
                                           .vulkanApiVersion = backend->api_version};

    i3_vk_check(vmaCreateAllocator(&allocator_ci, &device->vma));
    i3_log_dbg(device->log, "VMA created");

    // initialize resource pools
    i3_memory_pool_init(&device->use_list_block_pool, i3_alignof(i3_vk_use_list_block_t),
                        sizeof(i3_vk_use_list_block_t), I3_RESOURCE_BLOCK_CAPACITY);
    i3_memory_pool_init(&device->cmd_list_block_pool, i3_alignof(i3_vk_cmd_list_block_t),
                        sizeof(i3_vk_cmd_list_block_t), I3_VK_CMD_LIST_BLOCK_CAPACITY);
    i3_memory_pool_init(&device->sampler_pool, i3_alignof(i3_vk_sampler_o), sizeof(i3_vk_sampler_o),
                        I3_RESOURCE_BLOCK_CAPACITY);
    i3_memory_pool_init(&device->buffer_pool, i3_alignof(i3_vk_buffer_o), sizeof(i3_vk_buffer_o),
                        I3_RESOURCE_BLOCK_CAPACITY);
    i3_memory_pool_init(&device->image_pool, i3_alignof(i3_vk_image_o), sizeof(i3_vk_image_o),
                        I3_RESOURCE_BLOCK_CAPACITY);
    i3_memory_pool_init(&device->image_view_pool, i3_alignof(i3_vk_image_view_o), sizeof(i3_vk_image_view_o),
                        I3_RESOURCE_BLOCK_CAPACITY);
    i3_memory_pool_init(&device->descriptor_set_layout_pool, i3_alignof(i3_vk_descriptor_set_layout_o),
                        sizeof(i3_vk_descriptor_set_layout_o), I3_RESOURCE_BLOCK_CAPACITY);
    i3_memory_pool_init(&device->pipeline_layout_pool, i3_alignof(i3_vk_pipeline_layout_o),
                        sizeof(i3_vk_pipeline_layout_o), I3_RESOURCE_BLOCK_CAPACITY);
    i3_memory_pool_init(&device->framebuffer_pool, i3_alignof(i3_vk_framebuffer_o), sizeof(i3_vk_framebuffer_o),
                        I3_RESOURCE_BLOCK_CAPACITY);
    i3_memory_pool_init(&device->shader_module_pool, i3_alignof(i3_vk_shader_module_o), sizeof(i3_vk_shader_module_o),
                        I3_RESOURCE_BLOCK_CAPACITY);
    i3_memory_pool_init(&device->pipeline_pool, i3_alignof(i3_vk_pipeline_o), sizeof(i3_vk_pipeline_o),
                        I3_RESOURCE_BLOCK_CAPACITY);
    i3_memory_pool_init(&device->cmd_buffer_pool, i3_alignof(i3_vk_cmd_buffer_o), sizeof(i3_vk_cmd_buffer_o),
                        I3_RESOURCE_BLOCK_CAPACITY);
    i3_memory_pool_init(&device->submission_pool, i3_alignof(i3_vk_submission_t), sizeof(i3_vk_submission_t),
                        I3_RESOURCE_BLOCK_CAPACITY);

    // initialize submission array
    i3_array_init(&device->submissions, sizeof(i3_vk_submission_t*));

    i3_log_inf(device->log, "Vulkan device %s created", device_desc->base.name);

    return &device->iface;
}