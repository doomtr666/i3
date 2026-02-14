use ash::vk;
use i3_gfx::graph::backend::{BufferDesc, ImageDesc};
use vk_mem::{Alloc, Allocation, AllocationCreateInfo, Allocator, MemoryUsage};

pub struct VulkanImage {
    pub handle: vk::Image,
    pub allocation: Allocation,
    pub format: vk::Format,
}

impl VulkanImage {
    pub fn new(allocator: &Allocator, desc: &ImageDesc) -> Self {
        let format = crate::convert::to_vk_format(desc.format);
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: desc.width,
                height: desc.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let alloc_info = AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let (handle, allocation) = unsafe { allocator.create_image(&image_info, &alloc_info) }
            .expect("Failed to create VMA image");

        VulkanImage {
            handle,
            allocation,
            format,
        }
    }
}

pub struct VulkanBuffer {
    pub handle: vk::Buffer,
    pub allocation: Allocation,
}

impl VulkanBuffer {
    pub fn new(allocator: &Allocator, desc: &BufferDesc) -> Self {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(desc.size)
            .usage(
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let alloc_info = AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let (handle, allocation) = unsafe { allocator.create_buffer(&buffer_info, &alloc_info) }
            .expect("Failed to create VMA buffer");

        VulkanBuffer { handle, allocation }
    }
}
