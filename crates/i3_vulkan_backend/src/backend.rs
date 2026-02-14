use ash::vk;
use ash::vk::Handle;
use i3_gfx::graph::backend::*;
use i3_gfx::graph::types::*;
use std::ffi::CString;
use std::sync::Arc;
use tracing::info;

pub struct VulkanBackend {
    pub swapchain: Option<crate::swapchain::VulkanSwapchain>,
    pub window: Option<crate::window::VulkanWindow>,
    pub device: Arc<crate::device::VulkanDevice>,
    pub instance: Arc<crate::instance::VulkanInstance>,

    // Resource tracking for teardown
    pub pipelines: Vec<vk::Pipeline>,
    pub layouts: Vec<vk::PipelineLayout>,
    pub shader_modules: Vec<vk::ShaderModule>,
    pub image_acquired_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
}

impl VulkanBackend {
    pub fn new() -> Result<Self, String> {
        let instance = crate::instance::VulkanInstance::new()?;
        let device = Arc::new(crate::device::VulkanDevice::new(instance.clone())?);

        Ok(VulkanBackend {
            instance,
            device,
            window: None,
            swapchain: None,
            pipelines: Vec::new(),
            layouts: Vec::new(),
            shader_modules: Vec::new(),
            image_acquired_semaphores: Vec::new(),
            render_finished_semaphores: Vec::new(),
        })
    }
}

impl RenderBackend for VulkanBackend {
    fn create_image(&mut self, desc: &ImageDesc) -> BackendImage {
        let allocator = self.device.allocator.lock().unwrap();
        let image = crate::resources::VulkanImage::new(&allocator, desc);
        let ptr = Box::into_raw(Box::new(image));
        BackendImage(ptr as u64)
    }

    fn create_buffer(&mut self, desc: &BufferDesc) -> BackendBuffer {
        let allocator = self.device.allocator.lock().unwrap();
        let buffer = crate::resources::VulkanBuffer::new(&allocator, desc);
        let ptr = Box::into_raw(Box::new(buffer));
        BackendBuffer(ptr as u64)
    }

    fn create_graphics_pipeline(&mut self, desc: &GraphicsPipelineDesc) -> BackendPipeline {
        // For MVP, we'll implement a very minimal pipeline creation
        // In a real app we'd need more state (viewport, depth-stencil, etc.)
        // But Vulkan 1.3 Dynamic Rendering simplifies this.

        // Create a single shader module for the entire bytecode
        let module_info = vk::ShaderModuleCreateInfo::default().code(unsafe {
            std::slice::from_raw_parts(
                desc.shader.bytecode.as_ptr() as *const u32,
                desc.shader.bytecode.len() / 4,
            )
        });
        let module =
            unsafe { self.device.handle.create_shader_module(&module_info, None) }.unwrap();
        self.shader_modules.push(module);

        // Convert entry point names to CStrings and keep them alive
        let entry_points: Vec<CString> = desc
            .shader
            .stages
            .iter()
            .map(|s| CString::new(s.entry_point.as_str()).unwrap())
            .collect();

        let shader_stages: Vec<vk::PipelineShaderStageCreateInfo> = desc
            .shader
            .stages
            .iter()
            .enumerate()
            .map(|(i, stage)| {
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(crate::convert::to_vk_stage(stage.stage))
                    .module(module)
                    .name(entry_points[i].as_c_str())
            })
            .collect();

        // Minimal layout
        let layout_info = vk::PipelineLayoutCreateInfo::default();
        let layout = unsafe {
            self.device
                .handle
                .create_pipeline_layout(&layout_info, None)
        }
        .unwrap();
        self.layouts.push(layout);

        let ia_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let raster_state = vk::PipelineRasterizationStateCreateInfo::default().line_width(1.0);
        let ms_state = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA);
        let blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&blend_attachment));
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let color_formats = [vk::Format::B8G8R8A8_UNORM];
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

        let vi_state = vk::PipelineVertexInputStateCreateInfo::default();
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .layout(layout)
            .vertex_input_state(&vi_state)
            .input_assembly_state(&ia_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&raster_state)
            .multisample_state(&ms_state)
            .color_blend_state(&blend_state)
            .dynamic_state(&dynamic_state)
            .push_next(&mut rendering_info);

        let pipeline = unsafe {
            self.device.handle.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
        }
        .unwrap()[0];
        self.pipelines.push(pipeline);

        BackendPipeline(pipeline.as_raw())
    }

    fn create_swapchain(&mut self, _window_handle: u64, _usages: u32) -> u64 {
        // For the MVP, we assume the window is already created or we create it here
        // We'll use the window we have in the backend
        if let Some(ref window) = self.window {
            let width = window.handle.size().0;
            let height = window.handle.size().1;
            let sc = crate::swapchain::VulkanSwapchain::new(
                &self.instance.handle,
                &self.device.handle,
                self.device.physical_device,
                window.surface,
                width,
                height,
            )
            .unwrap();
            self.swapchain = Some(sc);
            0 // External handle not used yet
        } else {
            0
        }
    }

    fn present_swapchain(&mut self, _sc_handle: u64, _image: BackendImage) {
        // TODO: Implement presentation
    }

    fn begin_pass(&mut self, name: &str, f: Box<dyn FnOnce(&mut dyn PassContext) + Send + Sync>) {
        info!(pass = %name, "Beginning Vulkan pass");

        // Use a simple command pool/buffer for now
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.device.graphics_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { self.device.handle.create_command_pool(&pool_info, None) }.unwrap();

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buffer =
            unsafe { self.device.handle.allocate_command_buffers(&alloc_info) }.unwrap()[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .handle
                .begin_command_buffer(cmd_buffer, &begin_info)
        }
        .unwrap();

        // For dynamic rendering, we'd wrap this with vkCmdBeginRendering
        if let Some(ref window) = self.window {
            if let Some(ref mut swapchain) = self.swapchain {
                let semaphore_info = vk::SemaphoreCreateInfo::default();
                let acquire_sem =
                    unsafe { self.device.handle.create_semaphore(&semaphore_info, None) }.unwrap();
                let render_sem =
                    unsafe { self.device.handle.create_semaphore(&semaphore_info, None) }.unwrap();
                self.image_acquired_semaphores.push(acquire_sem);
                self.render_finished_semaphores.push(render_sem);

                let (image_index, _suboptimal) = unsafe {
                    swapchain
                        .loader
                        .acquire_next_image(
                            swapchain.handle,
                            u64::MAX,
                            acquire_sem,
                            vk::Fence::null(),
                        )
                        .unwrap()
                };

                let image = swapchain.images[image_index as usize];
                let image_view = swapchain.image_views[image_index as usize];
                let width = window.handle.size().0;
                let height = window.handle.size().1;

                // 1. Transition Image to Color Attachment Optimal
                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image(image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                let dependency_info = vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&barrier));

                // 2. Set dynamic state
                let viewport = vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: width as f32,
                    height: height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                let scissor = vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width, height },
                };

                unsafe {
                    self.device
                        .handle
                        .cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
                    self.device
                        .handle
                        .cmd_set_viewport(cmd_buffer, 0, &[viewport]);
                    self.device
                        .handle
                        .cmd_set_scissor(cmd_buffer, 0, &[scissor]);

                    let color_attach = vk::RenderingAttachmentInfo::default()
                        .image_view(image_view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.1, 0.2, 0.3, 1.0],
                            },
                        });

                    let rendering_info = vk::RenderingInfo::default()
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D { width, height },
                        })
                        .layer_count(1)
                        .color_attachments(std::slice::from_ref(&color_attach));

                    self.device
                        .handle
                        .cmd_begin_rendering(cmd_buffer, &rendering_info);
                }

                // 3. Execute Pass
                let mut ctx = crate::commands::VulkanPassContext {
                    device: &self.device.handle,
                    cmd_buffer,
                };
                f(&mut ctx);

                // 4. End Rendering and Transition to Present
                unsafe {
                    self.device.handle.cmd_end_rendering(cmd_buffer);

                    let barrier_present = vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .dst_access_mask(vk::AccessFlags2::NONE)
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .image(image)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        });

                    let dep_info_present = vk::DependencyInfo::default()
                        .image_memory_barriers(std::slice::from_ref(&barrier_present));

                    self.device
                        .handle
                        .cmd_pipeline_barrier2(cmd_buffer, &dep_info_present);
                    self.device.handle.end_command_buffer(cmd_buffer).unwrap();

                    // 5. Submit and Present
                    let cmd_buffers = [cmd_buffer];
                    let wait_semaphores = [acquire_sem];
                    let signal_semaphores = [render_sem];
                    let wait_dst_stage_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

                    let submit_info = vk::SubmitInfo::default()
                        .command_buffers(&cmd_buffers)
                        .wait_semaphores(&wait_semaphores)
                        .wait_dst_stage_mask(&wait_dst_stage_mask)
                        .signal_semaphores(&signal_semaphores);

                    self.device
                        .handle
                        .queue_submit(
                            self.device.graphics_queue,
                            &[submit_info],
                            vk::Fence::null(),
                        )
                        .unwrap();

                    let present_info = vk::PresentInfoKHR::default()
                        .swapchains(std::slice::from_ref(&swapchain.handle))
                        .image_indices(std::slice::from_ref(&image_index))
                        .wait_semaphores(&signal_semaphores);

                    swapchain
                        .loader
                        .queue_present(self.device.graphics_queue, &present_info)
                        .unwrap();
                    self.device.handle.device_wait_idle().unwrap();
                }
            }
        }

        unsafe {
            self.device.handle.destroy_command_pool(pool, None);
        }
    }

    fn resolve_image(&self, handle: ImageHandle) -> BackendImage {
        BackendImage(handle.0.0)
    }

    fn resolve_buffer(&self, handle: BufferHandle) -> BackendBuffer {
        BackendBuffer(handle.0.0)
    }

    fn resolve_pipeline(&self, handle: PipelineHandle) -> BackendPipeline {
        BackendPipeline(handle.0.0)
    }

    fn register_external_image(&mut self, _handle: ImageHandle, _physical: BackendImage) {}
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe {
            // Wait for all GPU work to finish
            let _ = self.device.handle.device_wait_idle();

            for &p in &self.pipelines {
                self.device.handle.destroy_pipeline(p, None);
            }
            for &l in &self.layouts {
                self.device.handle.destroy_pipeline_layout(l, None);
            }
            for &m in &self.shader_modules {
                self.device.handle.destroy_shader_module(m, None);
            }
            for &s in &self.image_acquired_semaphores {
                self.device.handle.destroy_semaphore(s, None);
            }
            for &s in &self.render_finished_semaphores {
                self.device.handle.destroy_semaphore(s, None);
            }
        }
        info!("Vulkan Backend destroyed (Pipelines, Layouts, Shaders cleaned)");
    }
}
