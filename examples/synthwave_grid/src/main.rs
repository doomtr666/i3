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

struct SynthwaveApp {
    backend: VulkanBackend,
    window: WindowHandle,
    pipeline: PipelineHandle,
    vertex_buffer: BackendBuffer,
    index_buffer: BackendBuffer,
    uniform_buffer: BackendBuffer,
    set_handle: DescriptorSetHandle,
    start_time: std::time::Instant,
}

impl ExampleApp for SynthwaveApp {
    fn update(&mut self, _delta: Duration) {
        let time = self.start_time.elapsed().as_secs_f32();

        // View Matrix (Camera at 0, 10, 50 looking at 0, 0, 0)
        // Simple LookAt for static camera
        // Right: (1, 0, 0)
        // Up: (0, 1, 0)
        // Forward: (0, 0, -1) (Looking down -Z)
        // Eye: (0, 10, 50)
        // View = Translate(-Eye) * Rotate(Identity for now, maybe title it down?)

        // Let's use a Orbit Camera logic or just fixed for now.
        // To see the grid from above/angle:
        // Position: (0, 20, 20)
        // Target: (0, 0, -20)
        // Up: (0, 1, 0)

        // Manual LookAt:
        // zaxis = normalize(eye - target)
        // xaxis = normalize(cross(up, zaxis))
        // yaxis = cross(zaxis, xaxis)
        // [ xaxis.x  xaxis.y  xaxis.z  -dot(xaxis, eye) ]
        // [ yaxis.x  yaxis.y  yaxis.z  -dot(yaxis, eye) ]
        // [ zaxis.x  zaxis.y  zaxis.z  -dot(zaxis, eye) ]
        // [ 0        0        0        1                ]

        // Hardcoding a simple camera for now:
        // Eye: (0, 10, 40)
        // Target: (0, 0, 0)
        // Up: (0, 1, 0)

        // Z (Forward) = (0, 10, 40) - (0,0,0) = (0, 10, 40) -> Normalized: (0, 0.24, 0.97)
        // X (Right) = (0, 1, 0) x Z = (1, 0, 0) NO wait. Cross up z.
        // Z is pointing TOWARDS viewer.
        // eye - target = vector from target to eye. Correct.
        // Z = (0, 10, 40). Length = sqrt(100 + 1600) = sqrt(1700) = 41.23
        // Z = (0, 0.242, 0.970)

        // X = Up x Z = (0, 1, 0) x (0, 0.242, 0.970) = (1*0.970 - 0*0.242, 0, 0) = (0.97? NO).
        // (0, 1, 0) x (0, y, z) = (z, 0, 0).
        // X = (0.970, 0, 0). Normalized -> (1, 0, 0).

        // Y = Z x X = (0, 0.242, 0.970) x (1, 0, 0) = (0, 0.970, -0.242).

        // Matrix:
        // 1 0 0 -Px
        // 0 0.970 0.242 -Py
        // 0 -0.242 0.970 -Pz
        // 0 0 0 1

        // Let's just use a fixed view matrix that looks slightly down.
        // Eye (0, 20, 40), looking at (0, 0, -10).

        // Projection (Perspective):
        // FovY = 45 deg = 0.785 rad.
        // Aspect = 1280/720 = 1.777.
        // Near = 0.1, Far = 1000.0.
        // f = 1.0 / tan(fov/2) = 1.0 / 0.414 = 2.414.
        // matrix:
        // f/aspect, 0, 0, 0
        // 0, -f, 0, 0  (Flip Y for Vulkan)
        // 0, 0, far/(near-far), -1
        // 0, 0, near*far/(near-far), 0

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
    }

    fn render(&mut self) {
        let mut graph = FrameGraph::new();
        let window = self.window;
        let pipeline = self.pipeline;
        let vb = self.vertex_buffer;
        let ib = self.index_buffer;
        let set = self.set_handle;

        graph.record(move |builder| {
            let backbuffer = builder.acquire_backbuffer(window);

            // Create Depth Buffer (Transient)
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

            builder.add_node("MainPass", move |sub| {
                sub.bind_pipeline(pipeline);
                sub.write_image(backbuffer, ResourceUsage::COLOR_ATTACHMENT);
                sub.write_image(depth_img, ResourceUsage::DEPTH_STENCIL);

                move |ctx| {
                    ctx.bind_vertex_buffer(0, BufferHandle(SymbolId(vb.0)));
                    ctx.bind_index_buffer(BufferHandle(SymbolId(ib.0)), IndexType::Uint16);
                    ctx.bind_descriptor_set(0, set);

                    ctx.draw_indexed(6, 0, 0);
                    ctx.present(backbuffer);
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

    // 4. Compile Shader
    let slang = SlangCompiler::new()?;
    let shader_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let shader =
        slang.compile_file("grid", ShaderTarget::Spirv, &[shader_dir.to_str().unwrap()])?;

    // 5. Create Pipeline
    let pipeline_info = GraphicsPipelineCreateInfo {
        shader_module: shader,
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
                    offset: 12, // size of float3
                },
            ],
        },
        render_targets: RenderTargetsInfo {
            color_targets: vec![RenderTargetInfo {
                format: Format::B8G8R8A8_SRGB,
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
    let backend_pipeline = backend.create_graphics_pipeline(&pipeline_info);
    let pipeline = PipelineHandle(SymbolId(backend_pipeline.0));

    // 6. Descriptor Set
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

    // 7. Run
    let app = SynthwaveApp {
        backend,
        window,
        pipeline,
        vertex_buffer,
        index_buffer,
        uniform_buffer,
        set_handle,
        start_time: std::time::Instant::now(),
    };
    main_loop(app);

    Ok(())
}
