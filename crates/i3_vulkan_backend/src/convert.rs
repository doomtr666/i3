use ash::vk;
use i3_gfx::graph::backend::ShaderStageFlags;

pub fn to_vk_stage(stage: ShaderStageFlags) -> vk::ShaderStageFlags {
    let mut flags = vk::ShaderStageFlags::empty();
    if stage.contains(ShaderStageFlags::Vertex) {
        flags |= vk::ShaderStageFlags::VERTEX;
    }
    if stage.contains(ShaderStageFlags::Fragment) {
        flags |= vk::ShaderStageFlags::FRAGMENT;
    }
    if stage.contains(ShaderStageFlags::Compute) {
        flags |= vk::ShaderStageFlags::COMPUTE;
    }
    flags
}

pub fn to_vk_format(format: u32) -> vk::Format {
    // Basic mapping for now
    match format {
        0 => vk::Format::B8G8R8A8_UNORM,
        1 => vk::Format::R8G8B8A8_UNORM,
        _ => vk::Format::UNDEFINED,
    }
}
