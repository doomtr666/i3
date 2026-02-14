use ash::vk;
use ash::vk::Handle;
use i3_gfx::graph::backend::*;
use i3_gfx::graph::types::*;

pub struct VulkanPassContext<'a> {
    pub device: &'a ash::Device,
    pub cmd_buffer: vk::CommandBuffer,
}

impl<'a> PassContext for VulkanPassContext<'a> {
    fn bind_pipeline(&mut self, pipeline: PipelineHandle) {
        let vk_pipeline = vk::Pipeline::from_raw(pipeline.0.0);
        unsafe {
            self.device.cmd_bind_pipeline(
                self.cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                vk_pipeline,
            );
        }
    }

    fn bind_image(&mut self, _slot: u32, _handle: ImageHandle) {
        // TODO: Bindless or descriptor sets
    }

    fn bind_buffer(&mut self, _slot: u32, _handle: BufferHandle) {
        // TODO: Bindless or descriptor sets
    }

    fn draw(&mut self, vertex_count: u32, first_vertex: u32) {
        unsafe {
            self.device
                .cmd_draw(self.cmd_buffer, vertex_count, 1, first_vertex, 0);
        }
    }

    fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        unsafe {
            self.device.cmd_dispatch(self.cmd_buffer, x, y, z);
        }
    }

    fn present(&mut self, _handle: ImageHandle) {
        // Handled by the backend after the pass
    }
}
