use examples_common::{ExampleApp, init_tracing, main_loop};
use i3_gfx::prelude::*;
use i3_slang::prelude::*;
use i3_vulkan_backend::VulkanBackend;
use nalgebra_glm as glm;
use std::time::Duration;

const SHADER_SOURCE: &str = r#"
struct PushConstants {
    float2 center;
    float zoom;
    float padding;
};

[[vk::push_constant]]
PushConstants pc;

RWTexture2D<float4> outputImage : register(u0);

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint width, height;
    outputImage.GetDimensions(width, height);
    
    if (threadId.x >= width || threadId.y >= height) return;

    float2 uv = float2(threadId.xy) / float2(width, height);
    float aspect = float(width) / float(height);
    
    float2 c = pc.center + (uv - 0.5) * float2(aspect, -1.0) * pc.zoom;
    
    float2 z = 0.0;
    int iter = 0;
    const int maxIter = 512;
    
    while (dot(z, z) < 4.0 && iter < maxIter) {
        z = float2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        iter++;
    }
    
    float3 color;
    if (iter == maxIter) {
        color = float3(0.02, 0.02, 0.05);
    } else {
        float sn = float(iter) - log2(log2(dot(z,z))) + 4.0;
        float f = sn / float(maxIter);
        color = 0.5 + 0.5 * cos(3.0 + f * 40.0 + float3(0.0, 0.6, 1.1));
        color *= pow(f, 0.1); 
    }
    
    outputImage[threadId.xy] = float4(color, 1.0);
}
"#;

/// Compute pass that renders the Mandelbrot set to a storage image.
struct MandelbrotPass {
    pipeline: PipelineHandle,
    backbuffer: ImageHandle,
    push_data: [f32; 4],
    width: u32,
    height: u32,
}

impl RenderPass for MandelbrotPass {
    fn name(&self) -> &str {
        "MandelbrotPass"
    }

    fn prefer_async(&self) -> bool {
        false
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        self.backbuffer = builder.resolve_image("Backbuffer");
        builder.bind_pipeline(self.pipeline);
        builder.write_image(self.backbuffer, ResourceUsage::SHADER_WRITE);
        builder.bind_descriptor_set(
            0,
            vec![DescriptorWrite {
                binding: 0,
                array_element: 0,
                descriptor_type: BindingType::StorageTexture,
                image_info: Some(DescriptorImageInfo {
                    image: self.backbuffer,
                    sampler: None,
                    image_layout: DescriptorImageLayout::General,
                }),
                buffer_info: None,
            }],
        );
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        ctx.push_constant_data(ShaderStageFlags::Compute, 0, &self.push_data);
        ctx.dispatch((self.width + 15) / 16, (self.height + 15) / 16, 1);
        ctx.present(self.backbuffer);
    }
}

struct MandelbrotApp {
    backend: VulkanBackend,
    window: WindowHandle,
    pass: MandelbrotPass,
    center: glm::Vec2,
    zoom: f32,
    mouse_pos: glm::Vec2,
    is_dragging: bool,
    width: u32,
    height: u32,
    is_fullscreen: bool,
}

impl MandelbrotApp {
    fn new() -> Self {
        let mut backend = VulkanBackend::new().unwrap();
        examples_common::maybe_list_gpus(&backend);
        backend
            .initialize(examples_common::get_gpu_index())
            .unwrap();

        let width = 1280;
        let height = 720;
        let window = backend
            .create_window(WindowDesc {
                title: "i3 - Compute Mandelbrot".to_string(),
                width,
                height,
            })
            .unwrap();

        backend
            .configure_window(
                window,
                SwapchainConfig {
                    vsync: false,
                    srgb: false,
                    min_image: 3,
                },
            )
            .unwrap();

        let compiler = SlangCompiler::new().unwrap();
        let shader_module = compiler
            .compile_inline(
                "mandelbrot_inline",
                "shader.slang",
                SHADER_SOURCE,
                ShaderTarget::Spirv,
            )
            .unwrap();

        let backend_pipeline =
            backend.create_compute_pipeline(&ComputePipelineCreateInfo { shader_module });

        let pass = MandelbrotPass {
            pipeline: PipelineHandle(SymbolId(backend_pipeline.0)),
            backbuffer: ImageHandle(SymbolId(0)),
            push_data: [0.0; 4],
            width,
            height,
        };

        Self {
            backend,
            window,
            pass,
            center: glm::vec2(-0.5, 0.0),
            zoom: 2.5,
            mouse_pos: glm::vec2(0.0, 0.0),
            is_dragging: false,
            width,
            height,
            is_fullscreen: false,
        }
    }
}

impl ExampleApp for MandelbrotApp {
    fn update(&mut self, _delta: Duration, _smoothed_delta: Duration) {}

    fn render(&mut self) {
        // Update pass state before recording
        self.pass.push_data = [self.center.x, self.center.y, self.zoom, 0.0];
        self.pass.width = self.width;
        self.pass.height = self.height;

        let mut graph = FrameGraph::new();
        let window = self.window;

        graph.record(|builder| {
            let backbuffer = builder.acquire_backbuffer(window);
            builder.publish("Backbuffer", backbuffer);
            builder.add_pass(&mut self.pass);
        });

        let compiled = graph.compile(&self.backend.capabilities());
        if let Err(e) = compiled.execute(&mut self.backend, None) {
            if e == GraphError::WindowMinimized {
                std::thread::sleep(Duration::from_millis(100));
            } else {
                panic!("Graph execution failed: {}", e);
            }
        }
    }

    fn poll_events(&mut self) -> Vec<Event> {
        self.backend.poll_events()
    }

    fn handle_event(&mut self, event: &Event) {
        match *event {
            Event::Resize { width, height, .. } => {
                self.width = width;
                self.height = height;
            }
            Event::MouseWheel { y, .. } => {
                let zoom_factor = if y > 0 { 0.95 } else { 1.05 };
                self.zoom *= zoom_factor;
            }
            Event::MouseDown { button: 1, .. } => {
                self.is_dragging = true;
            }
            Event::MouseUp { button: 1, .. } => {
                self.is_dragging = false;
            }
            Event::MouseMove { x, y, .. } => {
                let new_pos = glm::vec2(x as f32, y as f32);
                if self.is_dragging {
                    let delta = new_pos - self.mouse_pos;
                    let world_delta = delta * (self.zoom / (self.height as f32 / 2.0));
                    self.center.x -= world_delta.x;
                    self.center.y += world_delta.y;
                }
                self.mouse_pos = new_pos;
            }
            Event::KeyDown { key } => {
                if key == KeyCode::F11 {
                    self.is_fullscreen = !self.is_fullscreen;
                    self.backend.set_fullscreen(self.window, self.is_fullscreen);
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let _guard = init_tracing("compute_mandelbrot.log");
    let app = MandelbrotApp::new();
    main_loop(app);
}
