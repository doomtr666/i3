extern crate nalgebra_glm;
use examples_common::{ExampleApp, init_tracing, main_loop};
use i3_gfx::prelude::*;
use i3_slang::prelude::*;
use i3_vulkan_backend::VulkanBackend;
use nalgebra_glm as glm;
use std::time::Duration;
use tracing::{info, warn};

// Vertex Structure
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Vertex {
    pos: [f32; 3],
    uv: [f32; 2],
}

// Uniform Structure
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Uniforms {
    view_proj: glm::Mat4,
    time: f32,
    padding: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct PostUniforms {
    time: f32,
    resolution: [f32; 2],
    padding: f32,
}

struct SynthwaveApp {
    backend: VulkanBackend,
    window: WindowHandle,
    pipeline: PipelineHandle,
    vertex_buffer: BackendBuffer,
    index_buffer: BackendBuffer,
    uniform_buffer: BackendBuffer,
    set_handle: DescriptorSetHandle,

    // PostFX Resources
    post_pipeline: PipelineHandle,
    post_set: DescriptorSetHandle,
    scene_color: ImageHandle,
    #[allow(dead_code)]
    scene_sampler: SamplerHandle,
    post_uniform_buffer: BackendBuffer,

    start_time: std::time::Instant,
}

impl ExampleApp for SynthwaveApp {
    fn update(&mut self, _delta: Duration) {
        let time = self.start_time.elapsed().as_secs_f32();
        let aspect = 1280.0 / 720.0;
        let near = 0.1f32;
        let far = 1000.0f32;

        // Engine Convention: Right-Handed, Y-Up, Z[0, 1]
        // Vulkan Backend handles Viewport Flip (Negative Height)
        // so we use standard RH GLM functions.
        let proj = glm::perspective_rh_zo(aspect, 45.0f32.to_radians(), near, far);

        // Camera at (0, 20, 50), look at (0, 0, 0), up (0, 1, 0)
        let view = glm::look_at_rh(
            &glm::vec3(0.0, 20.0, 50.0),
            &glm::vec3(0.0, 0.0, 0.0),
            &glm::vec3(0.0, 1.0, 0.0),
        );

        // GLM matrices are Column-Major.
        // Proj * View is correct order for Column-Major P * V * v
        // Shader now explicitly expects column_major, so no transpose needed.
        let view_proj = proj * view;

        let uniforms = Uniforms {
            view_proj,
            time,
            padding: [0.0; 3],
        };

        let data_ptr = &uniforms as *const Uniforms as *const u8;
        let data_slice =
            unsafe { std::slice::from_raw_parts(data_ptr, std::mem::size_of::<Uniforms>()) };
        self.backend
            .upload_buffer(self.uniform_buffer, data_slice, 0)
            .unwrap();

        // Update Post Uniforms
        let post_uniforms = PostUniforms {
            time,
            resolution: [1280.0, 720.0],
            padding: 0.0,
        };
        let post_data_ptr = &post_uniforms as *const PostUniforms as *const u8;
        let post_data_slice = unsafe {
            std::slice::from_raw_parts(post_data_ptr, std::mem::size_of::<PostUniforms>())
        };
        self.backend
            .upload_buffer(self.post_uniform_buffer, post_data_slice, 0)
            .unwrap();
    }

    fn render(&mut self) {
        let mut graph = FrameGraph::new();
        let window = self.window;
        let pipeline = self.pipeline;
        let vb = self.vertex_buffer;
        let ib = self.index_buffer;
        let set = self.set_handle;

        // PostFX captures
        let post_pipeline = self.post_pipeline;
        let post_set = self.post_set;
        let scene_color = self.scene_color;
        // scene_sampler implicitly used in set
        // Actually we need to declare usage of scene_color in the second pass for Barriers!

        graph.record(move |builder| {
            let backbuffer = builder.acquire_backbuffer(window);

            // We use our registered external image logic for scene_color
            // But builder.import_image would be nice.
            // Since we used register_external_image, the backend knows it.
            // But the graph needs to know it to schedule barriers.
            // We can just use the handle we created "scene_color".

            // Create Depth Buffer (Transient) - Still needed for Grid
            let depth_desc = ImageDesc {
                width: 1280,
                height: 720,
                depth: 1,
                format: Format::D32_FLOAT,
                mip_levels: 1,
                array_layers: 1,
                usage: ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            };
            let depth_img = builder.declare_image("DepthBuffer", depth_desc);

            // Pass 1: Grid -> Scene Color
            builder.add_node("GridPass", move |sub| {
                sub.bind_pipeline(pipeline);
                sub.write_image(scene_color, ResourceUsage::COLOR_ATTACHMENT);
                sub.write_image(depth_img, ResourceUsage::DEPTH_STENCIL);

                move |ctx| {
                    ctx.bind_vertex_buffer(0, BufferHandle(SymbolId(vb.0)));
                    ctx.bind_index_buffer(BufferHandle(SymbolId(ib.0)), IndexType::Uint16);
                    ctx.bind_descriptor_set(0, set);

                    ctx.draw_indexed(6, 0, 0);
                }
            });

            // Pass 2: PostFX (Scene -> Backbuffer)
            builder.add_node("PostPass", move |sub| {
                sub.bind_pipeline(post_pipeline);
                sub.read_image(scene_color, ResourceUsage::SHADER_READ);
                sub.write_image(backbuffer, ResourceUsage::COLOR_ATTACHMENT);

                move |ctx| {
                    ctx.bind_descriptor_set(0, post_set);
                    ctx.draw(3, 0); // Fullscreen triangle (3 vertices)
                    ctx.present(backbuffer); // Present happens after this pass
                }
            });
        });

        let compiler = graph.compile();
        if let Err(e) = compiler.execute(&mut self.backend) {
            warn!("Graph execution failed: {}", e);
        }
    }
    fn poll_events(&mut self) -> Vec<Event> {
        self.backend.poll_events()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = init_tracing("synthwave_grid.log");
    info!("Starting Synthwave Grid Demo");

    // 1. Initialize Backend
    let mut backend = VulkanBackend::new()?;
    backend.initialize(0)?;

    // 2. Create Window
    let window = backend.create_window(WindowDesc {
        title: "Synthwave Grid".to_string(),
        width: 1280,
        height: 720,
    })?;

    // 3. Create Resources
    let vertices = [
        Vertex {
            pos: [-50.0, 0.0, -50.0],
            uv: [0.0, 0.0],
        },
        Vertex {
            pos: [50.0, 0.0, -50.0],
            uv: [100.0, 0.0],
        },
        Vertex {
            pos: [-50.0, 0.0, 50.0],
            uv: [0.0, 100.0],
        },
        Vertex {
            pos: [50.0, 0.0, 50.0],
            uv: [100.0, 100.0],
        },
    ];

    // Explicit list:
    let indices: [u16; 6] = [0, 2, 1, 1, 2, 3];

    let vb_desc = BufferDesc {
        size: std::mem::size_of_val(&vertices) as u64,
        usage: BufferUsageFlags::VERTEX_BUFFER,
        memory: MemoryType::CpuToGpu,
    };
    let vertex_buffer = backend.create_buffer(&vb_desc);

    let data_ptr = vertices.as_ptr() as *const u8;
    let data_slice =
        unsafe { std::slice::from_raw_parts(data_ptr, std::mem::size_of_val(&vertices)) };
    backend.upload_buffer(vertex_buffer, data_slice, 0)?;

    let ib_desc = BufferDesc {
        size: std::mem::size_of_val(&indices) as u64,
        usage: BufferUsageFlags::INDEX_BUFFER,
        memory: MemoryType::CpuToGpu,
    };
    let index_buffer = backend.create_buffer(&ib_desc);

    let data_ptr = indices.as_ptr() as *const u8;
    let data_slice =
        unsafe { std::slice::from_raw_parts(data_ptr, std::mem::size_of_val(&indices)) };
    backend.upload_buffer(index_buffer, data_slice, 0)?;

    let ub_desc = BufferDesc {
        size: std::mem::size_of::<Uniforms>() as u64,
        usage: BufferUsageFlags::UNIFORM_BUFFER,
        memory: MemoryType::CpuToGpu,
    };
    let uniform_buffer = backend.create_buffer(&ub_desc);

    // Post Uniform Buffer
    let pub_desc = BufferDesc {
        size: std::mem::size_of::<PostUniforms>() as u64,
        usage: BufferUsageFlags::UNIFORM_BUFFER | BufferUsageFlags::TRANSFER_DST,
        memory: MemoryType::CpuToGpu,
    };
    let post_uniform_buffer = backend.create_buffer(&pub_desc);

    // Scene Color & Sampler
    let scene_desc = ImageDesc {
        width: 1280,
        height: 720,
        depth: 1,
        format: Format::B8G8R8A8_SRGB,
        mip_levels: 1,
        array_layers: 1,
        usage: ImageUsageFlags::COLOR_ATTACHMENT
            | ImageUsageFlags::SAMPLED
            | ImageUsageFlags::TRANSFER_SRC,
    };
    let scene_physical = backend.create_image(&scene_desc);
    let scene_color = ImageHandle(SymbolId(10000)); // Persistent Handle
    backend.register_external_image(scene_color, scene_physical);

    let sampler_desc = SamplerDesc::default();
    let scene_sampler = backend.create_sampler(&sampler_desc);

    // 4. Compile Shaders
    let slang = SlangCompiler::new()?;
    let shader_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let grid_shader =
        slang.compile_file("grid", ShaderTarget::Spirv, &[shader_dir.to_str().unwrap()])?;
    let crt_shader =
        slang.compile_file("crt", ShaderTarget::Spirv, &[shader_dir.to_str().unwrap()])?;

    // 5. Create Pipelines
    // Grid Pipeline
    let grid_pipeline_info = GraphicsPipelineCreateInfo {
        shader_module: grid_shader,
        vertex_input: VertexInputState {
            bindings: vec![VertexInputBinding {
                binding: 0,
                stride: std::mem::size_of::<Vertex>() as u32,
                input_rate: VertexInputRate::Vertex,
            }],
            attributes: vec![
                VertexInputAttribute {
                    location: 0,
                    binding: 0,
                    format: VertexFormat::Float3,
                    offset: 0,
                },
                VertexInputAttribute {
                    location: 1,
                    binding: 0,
                    format: VertexFormat::Float2,
                    offset: 12,
                },
            ],
        },
        render_targets: RenderTargetsInfo {
            color_targets: vec![RenderTargetInfo {
                format: Format::B8G8R8A8_SRGB, // SceneColor
                ..Default::default()
            }],
            depth_stencil_format: Some(Format::D32_FLOAT),
        },
        input_assembly: InputAssemblyState {
            topology: PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        rasterization_state: RasterizationState {
            cull_mode: CullMode::None,
            ..Default::default()
        },
        ..Default::default()
    };
    let backend_grid = backend.create_graphics_pipeline(&grid_pipeline_info);
    let pipeline = PipelineHandle(SymbolId(backend_grid.0));

    // CRT Pipeline
    let crt_pipeline_info = GraphicsPipelineCreateInfo {
        shader_module: crt_shader,
        vertex_input: VertexInputState::default(),
        render_targets: RenderTargetsInfo {
            color_targets: vec![RenderTargetInfo {
                format: Format::B8G8R8A8_SRGB, // Backbuffer
                ..Default::default()
            }],
            depth_stencil_format: None,
        },
        rasterization_state: RasterizationState {
            cull_mode: CullMode::None,
            front_face: FrontFace::CounterClockwise,
            ..Default::default()
        },
        ..Default::default()
    };
    let backend_crt = backend.create_graphics_pipeline(&crt_pipeline_info);
    let post_pipeline = PipelineHandle(SymbolId(backend_crt.0));

    // 6. Descriptor Sets
    let set_handle = backend.allocate_descriptor_set(pipeline, 0)?;
    backend.update_descriptor_set(
        set_handle,
        &[DescriptorWrite {
            binding: 0,
            array_element: 0,
            descriptor_type: BindingType::UniformBuffer,
            buffer_info: Some(DescriptorBufferInfo {
                buffer: BufferHandle(SymbolId(uniform_buffer.0)),
                offset: 0,
                range: std::mem::size_of::<Uniforms>() as u64,
            }),
            image_info: None,
        }],
    );

    let post_set = backend.allocate_descriptor_set(post_pipeline, 0)?;
    backend.update_descriptor_set(
        post_set,
        &[
            DescriptorWrite {
                binding: 0,
                array_element: 0,
                descriptor_type: BindingType::UniformBuffer,
                buffer_info: Some(DescriptorBufferInfo {
                    buffer: BufferHandle(SymbolId(post_uniform_buffer.0)),
                    offset: 0,
                    range: std::mem::size_of::<PostUniforms>() as u64,
                }),
                image_info: None,
            },
            DescriptorWrite {
                binding: 1,
                array_element: 0,
                descriptor_type: BindingType::CombinedImageSampler,
                buffer_info: None,
                image_info: Some(DescriptorImageInfo {
                    image: scene_color,
                    image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                    sampler: Some(scene_sampler),
                }),
            },
        ],
    );

    // 7. Run
    let app = SynthwaveApp {
        backend,
        window,
        pipeline,
        vertex_buffer,
        index_buffer,
        uniform_buffer,
        set_handle,

        post_pipeline,
        post_set,
        scene_color,
        scene_sampler,
        post_uniform_buffer,

        start_time: std::time::Instant::now(),
    };
    main_loop(app);

    Ok(())
}
