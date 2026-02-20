use examples_common::{ExampleApp, init_tracing, main_loop};
use i3_gfx::graph::backend::{
    CommandBatch, DescriptorImageLayout, DescriptorWrite, Event, PassDescriptor, RenderBackend,
    ResourceUsage, SwapchainConfig, WindowDesc,
};
use i3_gfx::graph::pipeline::{BindingType, ComputePipelineCreateInfo, ShaderStageFlags};
use i3_gfx::graph::types::{ImageHandle, PipelineHandle, SymbolId, WindowHandle};
use i3_slang::{ShaderTarget, SlangCompiler};
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
        color = float3(0.02, 0.02, 0.05); // Deep dark blue for the set
    } else {
        // Smooth coloring (Normalized Escape Time)
        float sn = float(iter) - log2(log2(dot(z,z))) + 4.0;
        float f = sn / float(maxIter);
        
        // Premium Vibrant Palette (Cosine-based)
        color = 0.5 + 0.5 * cos(3.0 + f * 40.0 + float3(0.0, 0.6, 1.1));
        
        // Darken the edges slightly
        color *= pow(f, 0.1); 
    }
    
    outputImage[threadId.xy] = float4(color, 1.0);
}
"#;

struct MandelbrotApp {
    backend: Box<dyn RenderBackend>,
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
        let mut backend = i3_vulkan_backend::VulkanBackend::new()
            .map(|b| Box::new(b) as Box<dyn RenderBackend>)
            .unwrap();
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

        // Compile Shader
        let compiler = SlangCompiler::new().unwrap();
        let shader_module = compiler
            .compile_inline(
                "mandelbrot_inline",
                "shader.slang",
                SHADER_SOURCE,
                ShaderTarget::Spirv,
            )
            .unwrap();

        let pipeline =
            backend.create_compute_pipeline(&ComputePipelineCreateInfo { shader_module });

        Self {
            backend,
            window,
            pipeline: PipelineHandle(SymbolId(pipeline.0)),
            center: glm::vec2(-0.5, 0.0),
            zoom: 2.5,
            mouse_pos: glm::vec2(0.0, 0.0),
            is_dragging: false,
            width,
            height,
            // descriptor_set: None,
        }
    }
}

impl ExampleApp for MandelbrotApp {
    fn update(&mut self, _delta: Duration) {}

    fn render(&mut self) {
        self.backend.begin_frame();

        if let Ok(Some((swap_image, semaphore_val, _))) =
            self.backend.acquire_swapchain_image(self.window)
        {
            let swap_handle = ImageHandle(SymbolId(999));
            self.backend
                .register_external_image(swap_handle, swap_image);

            let pipeline_handle = self.pipeline;

            // let set = if let Some(s) = self.descriptor_set { ... } // Replaced by declarative bindings

            let push_data = [
                self.center.x,
                self.center.y,
                self.zoom,
                0.0, // padding
            ];
            let push_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(push_data.as_ptr() as *const u8, push_data.len() * 4)
            };

            let pass_name = "MandelbrotPass";
            let image_writes = [(swap_handle, ResourceUsage::SHADER_WRITE)];

            self.backend.begin_pass(
                PassDescriptor {
                    name: pass_name,
                    pipeline: Some(pipeline_handle),
                    image_reads: &[],
                    image_writes: &image_writes,
                    buffer_reads: &[],
                    buffer_writes: &[],
                    descriptor_sets: &[(
                        0,
                        vec![DescriptorWrite {
                            binding: 0,
                            array_element: 0,
                            descriptor_type: BindingType::StorageTexture,
                            image_info: Some(i3_gfx::graph::backend::DescriptorImageInfo {
                                image: swap_handle,
                                sampler: None,
                                image_layout: DescriptorImageLayout::General,
                            }),
                            buffer_info: None,
                        }],
                    )],
                },
                Box::new(move |ctx| {
                    ctx.bind_pipeline(pipeline_handle);
                    // ctx.bind_descriptor_set(0, set); // Handled by begin_pass
                    ctx.push_constants(ShaderStageFlags::Compute, 0, push_bytes);
                    ctx.dispatch((1280 + 15) / 16, (720 + 15) / 16, 1);
                    ctx.present(swap_handle);
                }),
            );

            let batch = CommandBatch::default();

            let _ = self.backend.submit(batch, &[semaphore_val], &[]).unwrap();
        }

        self.backend.end_frame();
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
