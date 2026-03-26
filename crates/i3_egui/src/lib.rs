pub mod input;
pub mod renderer;
pub mod prelude;

use std::sync::Arc;
use i3_gfx::graph::backend::RenderBackend;
use std::sync::atomic::{AtomicBool, Ordering};
pub use egui;

pub struct UiSystem {
    ctx: egui::Context,
    renderer: Arc<std::sync::Mutex<renderer::EguiRenderer>>,
    start_time: std::time::Instant,
    stored_output: Arc<std::sync::Mutex<Option<egui::FullOutput>>>,
    begun: AtomicBool,
    
    // UI Controller state
    events_buffer: Arc<std::sync::Mutex<Vec<i3_gfx::graph::backend::Event>>>,
    screen_size: Arc<std::sync::Mutex<(u32, u32)>>,
}

impl UiSystem {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            ctx: egui::Context::default(),
            renderer: Arc::new(std::sync::Mutex::new(renderer::EguiRenderer::new())),
            start_time: std::time::Instant::now(),
            stored_output: Arc::new(std::sync::Mutex::new(None)),
            begun: AtomicBool::new(false),
            events_buffer: Arc::new(std::sync::Mutex::new(Vec::new())),
            screen_size: Arc::new(std::sync::Mutex::new((width, height))),
        }
    }

    pub fn init_from_baked(&self, backend: &mut dyn RenderBackend, asset: &i3_io::pipeline_asset::PipelineAsset) {
        self.renderer.lock().unwrap().init_from_baked(backend, asset);
    }

    pub fn context(&self) -> &egui::Context {
        &self.ctx
    }

    pub fn handle_event(&self, event: &i3_gfx::graph::backend::Event) {
        let mut events = self.events_buffer.lock().unwrap();
        events.push(event.clone());
        
        // Immediate update of screen size for responsiveness
        if let i3_gfx::graph::backend::Event::Resize { width, height } = event {
            let mut size = self.screen_size.lock().unwrap();
            *size = (*width, *height);
        }
    }

    pub fn begin_frame(&self) {
        self.begun.store(true, Ordering::SeqCst);
        
        let mut events = self.events_buffer.lock().unwrap();
        let (width, height) = *self.screen_size.lock().unwrap();
        
        let raw_input = egui::RawInput {
            time: Some(self.start_time.elapsed().as_secs_f64()),
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(width as f32, height as f32),
            )),
            events: events.drain(..).filter_map(|e| input::map_event(&e)).collect(),
            ..Default::default()
        };
        
        self.ctx.begin_pass(raw_input);
    }

    pub fn update_textures(&self, backend: &mut dyn RenderBackend) {
        if !self.begun.swap(false, Ordering::SeqCst) {
            return;
        }
        let full_output = self.ctx.end_pass();
        self.renderer.lock().unwrap().update_textures(backend, &full_output.textures_delta);
        // Store the output for the record phase
        let mut storage = self.stored_output.lock().unwrap();
        *storage = Some(full_output);
    }

    pub fn create_pass(&self, backbuffer: i3_gfx::graph::types::ImageHandle) -> Option<renderer::EguiPass> {
        let (width, height) = *self.screen_size.lock().unwrap();
        let mut storage = self.stored_output.lock().unwrap();
        
        if let Some(full_output) = storage.take() {
            let ppp = full_output.pixels_per_point;
            let primitives = self.ctx.tessellate(full_output.shapes, ppp); 
            Some(renderer::EguiPass::new(self.renderer.clone(), primitives, width, height, backbuffer))
        } else {
            // Return an empty pass for initialization purposes if no output is pending
            Some(renderer::EguiPass::new(self.renderer.clone(), Vec::new(), width, height, backbuffer))
        }
    }
}
