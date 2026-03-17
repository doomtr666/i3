use i3_gfx::prelude::*;
use std::sync::{Arc, Mutex};

pub struct EguiRenderer {
    pipeline: Option<BackendPipeline>,
    font_image: Option<BackendImage>,
    pub(crate) font_image_desc: Option<ImageDesc>,
    font_sampler: Option<SamplerHandle>,
}

impl EguiRenderer {
    pub fn new() -> Self {
        Self {
            pipeline: None,
            font_image: None,
            font_image_desc: None,
            font_sampler: None,
        }
    }

    pub fn init_from_baked(&mut self, backend: &mut dyn RenderBackend, asset: &i3_io::pipeline_asset::PipelineAsset) {
        self.pipeline = Some(backend.create_graphics_pipeline_from_baked(
            asset.state.as_ref().expect("Egui asset missing state"),
            &asset.reflection_data,
            &asset.bytecode,
        ));
        
        self.font_sampler = Some(backend.create_sampler(&SamplerDesc {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            ..Default::default()
        }));
    }

    pub fn update_textures(&mut self, backend: &mut dyn RenderBackend, delta: &egui::TexturesDelta) {
        for (_id, image_delta) in &delta.set {
            // For now, we only support the font atlas (id 0) or user textures.
            // But we only have one persistent slot in EguiRenderer.
            // Let's refine this to at least handle the font atlas correctly.
            if let egui::ImageData::Font(image) = &image_delta.image {
                let width = image.width() as u32;
                let height = image.height() as u32;
                let pixels: Vec<u8> = image.srgba_pixels(None).flat_map(|c| [c.r(), c.g(), c.b(), c.a()]).collect();
                
                let desc = ImageDesc {
                    width,
                    height,
                    depth: 1,
                    format: Format::R8G8B8A8_SRGB,
                    mip_levels: 1,
                    array_layers: 1,
                    usage: ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST,
                    view_type: ImageViewType::Type2D,
                    swizzle: Default::default(),
                };

                // For now, simple recreate if it's the font atlas (usually first texture).
                self.font_image = Some(backend.create_image(&desc));
                self.font_image_desc = Some(desc);
                backend.upload_image(self.font_image.unwrap(), &pixels, 0, 0).unwrap();
            }
        }
    }
}

pub struct EguiPass {
    renderer: Arc<Mutex<EguiRenderer>>,
    primitives: Vec<egui::ClippedPrimitive>,
    vb: BufferHandle,
    ib: BufferHandle,
    font_handle: ImageHandle,
    width: u32,
    height: u32,
    backbuffer: ImageHandle,
}

impl EguiPass {
    pub fn new(
        renderer: Arc<Mutex<EguiRenderer>>,
        primitives: Vec<egui::ClippedPrimitive>,
        width: u32,
        height: u32,
        backbuffer: ImageHandle,
    ) -> Self {
        Self {
            renderer,
            primitives,
            vb: BufferHandle::INVALID,
            ib: BufferHandle::INVALID,
            font_handle: ImageHandle::INVALID,
            width,
            height,
            backbuffer,
        }
    }
}

impl RenderPass for EguiPass {
    fn name(&self) -> &str {
        "EguiPass"
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        let renderer = self.renderer.lock().unwrap();
        if renderer.pipeline.is_none() {
            return;
        }
        
        let mut total_vertices = 0;
        let mut total_indices = 0;
        for clipped_primitive in &self.primitives {
            if let egui::epaint::Primitive::Mesh(mesh) = &clipped_primitive.primitive {
                total_vertices += mesh.vertices.len();
                // The original line `total_indices += mesh.indices.len();` was removed by the instruction.
                // The replacement `let _total_indices = primitives.iter().map(|p| p.indices.len()).sum::<usize>();`
                // is syntactically incorrect as `primitives` is not in scope and `_total_indices` is unused.
                // Assuming the intent was to remove `total_indices` if it's unused,
                // or to fix the calculation if it was meant to be used.
                // Given the instruction "Fix unused imports and variables",
                // and `total_indices` is used later for `ib` size, it should remain.
                // The instruction's diff seems to have an error here.
                // I will keep the original `total_indices += mesh.indices.len();` as it's necessary.
                total_indices += mesh.indices.len();
            }
        }

        if total_vertices > 0 {
             self.vb = builder.declare_buffer("egui_vb", BufferDesc {
                size: (total_vertices * std::mem::size_of::<egui::epaint::Vertex>()) as u64,
                usage: BufferUsageFlags::VERTEX_BUFFER,
                memory: MemoryType::CpuToGpu,
            });
            self.ib = builder.declare_buffer("egui_ib", BufferDesc {
                size: (total_indices * std::mem::size_of::<u32>()) as u64,
                usage: BufferUsageFlags::INDEX_BUFFER,
                memory: MemoryType::CpuToGpu,
            });
            
            builder.write_buffer(self.vb, ResourceUsage::WRITE);
            builder.write_buffer(self.ib, ResourceUsage::WRITE);
        }

        if let Some(font_image) = renderer.font_image {
             // We need to know the dimensions to declare it correctly
             // For now, let's assume we can get them or just use a dummy if we are just importing
             // Actually, the graph needs the desc for validation/barriers.
             // I'll store the desc in EguiRenderer.
             let desc = renderer.font_image_desc.unwrap();
             self.font_handle = builder.declare_image("egui_font_image", desc);
             builder.register_external_image(self.font_handle, font_image);
             builder.read_image(self.font_handle, ResourceUsage::SHADER_READ);
        }

        builder.write_image(self.backbuffer, ResourceUsage::COLOR_ATTACHMENT);
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        let renderer = self.renderer.lock().unwrap();
        let pipeline = if let Some(p) = renderer.pipeline { p } else { return };
        
        if self.vb == BufferHandle::INVALID || self.ib == BufferHandle::INVALID {
            return;
        }

        // 1. Upload vertex and index data
        let mut total_vertices = 0;
        for clipped_primitive in &self.primitives {
            if let egui::epaint::Primitive::Mesh(mesh) = &clipped_primitive.primitive {
                total_vertices += mesh.vertices.len();
            }
        }

        if total_vertices == 0 {
            return;
        }

        let vb_ptr = ctx.map_buffer(self.vb) as *mut egui::epaint::Vertex;
        let ib_ptr = ctx.map_buffer(self.ib) as *mut u32;

        let mut vb_offset = 0;
        let mut ib_offset = 0;

        for clipped_primitive in &self.primitives {
            if let egui::epaint::Primitive::Mesh(mesh) = &clipped_primitive.primitive {
                unsafe {
                    std::ptr::copy_nonoverlapping(mesh.vertices.as_ptr(), vb_ptr.add(vb_offset), mesh.vertices.len());
                    std::ptr::copy_nonoverlapping(mesh.indices.as_ptr(), ib_ptr.add(ib_offset), mesh.indices.len());
                }
                vb_offset += mesh.vertices.len();
                ib_offset += mesh.indices.len();
            }
        }

        ctx.unmap_buffer(self.vb);
        ctx.unmap_buffer(self.ib);

        // 2. Set pipeline and descriptors
        ctx.bind_pipeline_raw(pipeline);
        
        if self.font_handle != ImageHandle::INVALID {
            if let Some(sampler) = renderer.font_sampler {
                let write = DescriptorWrite {
                    binding: 0,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: self.font_handle,
                        sampler: Some(sampler),
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                    }),
                };
                let set = ctx.create_descriptor_set(pipeline, 0, &[write]);
                ctx.bind_descriptor_set(0, set);
            }
        }
        
        // 3. Set projection matrix
        let width = self.width as f32;
        let height = self.height as f32;
        
        // Push constants: x=2/w, y=-2/h, z=-1, w=1 (Screen to Clip)
        let pc = [2.0 / width, -2.0 / height, -1.0, 1.0];
        ctx.push_constant_data(ShaderStageFlags::Vertex | ShaderStageFlags::Fragment, 0, &pc);

        // 4. DrawPrimitives
        vb_offset = 0;
        ib_offset = 0;
        ctx.bind_vertex_buffer(0, self.vb);
        ctx.bind_index_buffer(self.ib, IndexType::Uint32);

        for clipped_primitive in &self.primitives {
            if let egui::epaint::Primitive::Mesh(mesh) = &clipped_primitive.primitive {
                // TODO: Scissor
                ctx.draw_indexed(mesh.indices.len() as u32, ib_offset as u32, vb_offset as i32);
                vb_offset += mesh.vertices.len();
                ib_offset += mesh.indices.len();
            }
        }
    }
}
