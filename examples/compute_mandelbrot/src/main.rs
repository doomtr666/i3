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

struct MandelbrotApp {
    backend: VulkanBackend,
    window: WindowHandle,
    pipeline: PipelineHandle,
    center: glm::Vec2,
    zoom: f32,
    mouse_pos: glm::Vec2,
    is_dragging: bool,
    width: u32,
    height: u32,
}

impl MandelbrotApp {
    fn new() -> Self {
        let mut backend = VulkanBackend::new().unwrap();
        backend.initialize(0).unwrap();

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

        Self {
            backend,
            window,
            pipeline: PipelineHandle(SymbolId(backend_pipeline.0)),
            center: glm::vec2(-0.5, 0.0),
            zoom: 2.5,
            mouse_pos: glm::vec2(0.0, 0.0),
            is_dragging: false,
            width,
            height,
        }
    }
}

impl ExampleApp for MandelbrotApp {
    fn update(&mut self, _delta: Duration) {}

    fn render(&mut self) {
        let mut graph = FrameGraph::new();
        let window = self.window;
        let pipeline = self.pipeline;
        let width = self.width;
        let height = self.height;

        let push_data = [self.center.x, self.center.y, self.zoom, 0.0];
        let push_bytes: Vec<u8> = push_data.iter().flat_map(|f| f.to_ne_bytes()).collect();

        graph.record(move |builder| {
            let backbuffer = builder.acquire_backbuffer(window);

            builder.add_node("MandelbrotPass", move |sub| {
                sub.bind_pipeline(pipeline);
                sub.write_image(backbuffer, ResourceUsage::SHADER_WRITE);
                sub.bind_descriptor_set(
                    0,
                    vec![DescriptorWrite {
                        binding: 0,
                        array_element: 0,
                        descriptor_type: BindingType::StorageTexture,
                        image_info: Some(DescriptorImageInfo {
                            image: backbuffer,
                            sampler: None,
                            image_layout: DescriptorImageLayout::General,
                        }),
                        buffer_info: None,
                    }],
                );

                move |ctx| {
                    ctx.push_constants(ShaderStageFlags::Compute, 0, &push_bytes);
                    ctx.dispatch((width + 15) / 16, (height + 15) / 16, 1);
                    ctx.present(backbuffer);
                }
            });
        });

        let compiled = graph.compile();
        match compiled.execute(&mut self.backend) {
            Ok(_) => {}
            Err(e) if e == "WindowMinimized" => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => panic!("Graph execution failed: {}", e),
        }
    }

    fn poll_events(&mut self) -> Vec<Event> {
        let events = self.backend.poll_events();
        for event in &events {
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
                _ => {}
            }
        }
        events
    }
}

fn main() {
    let _guard = init_tracing("compute_mandelbrot.log");
    let app = MandelbrotApp::new();
    main_loop(app);
}
