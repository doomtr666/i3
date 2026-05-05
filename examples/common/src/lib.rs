pub mod basic_scene;
pub mod camera_controller;

use i3_gfx::prelude::{RenderBackend, WindowDesc, WindowHandle};
use i3_io::asset::AssetLoader;
use i3_renderer::prelude::DefaultRenderGraph;
use i3_renderer::render_graph::RenderConfig;
use i3_vulkan_backend::backend::VulkanBackend;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::info;

// ─── Renderer bootstrap ──────────────────────────────────────────────────────

pub struct AppRenderer {
    pub backend:      VulkanBackend,
    pub window:       WindowHandle,
    pub render_graph: DefaultRenderGraph,
    pub ui:           Arc<i3_egui::UiSystem>,
}

/// Initialise backend, window, render graph, and egui in one call.
///
/// The caller is responsible for creating the `AssetLoader` (with whatever VFS
/// mounts it needs) and passing it in — this lets each example control its own
/// asset pipeline while sharing the common setup boilerplate.
pub fn init_renderer(
    title: &str,
    width: u32,
    height: u32,
    loader: Option<Arc<AssetLoader>>,
) -> Result<AppRenderer, Box<dyn std::error::Error>> {
    let mut backend = VulkanBackend::new()?;
    maybe_list_gpus(&backend);
    backend.initialize(get_gpu_index())?;

    let window = backend.create_window(WindowDesc {
        title: title.to_string(),
        width,
        height,
    })?;

    let config = RenderConfig { width, height };
    let ui = Arc::new(i3_egui::UiSystem::new(width, height));

    let mut render_graph = DefaultRenderGraph::new(&mut backend, &config);
    render_graph.publish("UiSystem", ui.clone());
    if let Some(l) = loader {
        render_graph.publish("AssetLoader", l);
    }
    render_graph.init(&mut backend);

    Ok(AppRenderer { backend, window, render_graph, ui })
}
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

pub fn init_tracing(file_name: &str) -> tracing_appender::non_blocking::WorkerGuard {
    let mut level = "info";
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "-v" || arg == "--verbose") {
        level = "debug";
    }

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(format!(
            "info,i3_vulkan_backend={},i3_gfx={},i3_io={},i3_baker={},i3_renderer={},viewer={}",
            level, level, level, level, level, level
        ))
    });

    // Ensure logs directory exists
    let _ = std::fs::create_dir("logs");

    // Create (truncate) the log file
    let file =
        std::fs::File::create(format!("logs/{}", file_name)).expect("Failed to create log file");

    let (non_blocking, guard) = tracing_appender::non_blocking(file);

    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(non_blocking))
        .with(fmt::layer()) // Also to stdout
        .with(filter)
        .init();

    guard
}

pub struct FrameStats {
    last_frame: Instant,
    frame_count: u32,
    accumulated_time: Duration,
    displayed_dt: Duration,
}

impl Default for FrameStats {
    fn default() -> Self {
        Self {
            last_frame: Instant::now(),
            frame_count: 0,
            accumulated_time: Duration::ZERO,
            displayed_dt: Duration::from_millis(16),
        }
    }
}

impl FrameStats {
    pub fn update(&mut self) -> Duration {
        let now = Instant::now();
        let delta = now.duration_since(self.last_frame);
        self.last_frame = now;

        self.frame_count += 1;
        self.accumulated_time += delta;

        // Update the displayed average every 300ms for readability
        if self.accumulated_time.as_secs_f32() >= 0.3 {
            self.displayed_dt = self.accumulated_time / self.frame_count;
            self.frame_count = 0;
            self.accumulated_time = Duration::ZERO;
        }

        delta
    }

    pub fn smoothed_dt(&self) -> Duration {
        self.displayed_dt
    }
}

pub trait ExampleApp {
    fn update(&mut self, delta: Duration, smoothed_delta: Duration);
    fn render(&mut self);
    fn poll_events(&mut self) -> Vec<i3_gfx::graph::backend::Event>;
    fn handle_event(&mut self, event: &i3_gfx::graph::backend::Event);
}

pub fn get_gpu_index() -> u32 {
    let args: Vec<String> = std::env::args().collect();
    for i in 0..args.len() {
        if (args[i] == "--gpu" || args[i] == "-g") && i + 1 < args.len() {
            if let Ok(index) = args[i + 1].parse::<u32>() {
                return index;
            }
        }
    }
    0
}

pub fn maybe_list_gpus(backend: &dyn i3_gfx::graph::backend::RenderBackend) {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--list-gpus" || arg == "-l") {
        let devices = backend.enumerate_devices();
        println!("\nAvailable GPUs:");
        for dev in devices {
            println!("  [{}] {} ({:?})", dev.id, dev.name, dev.device_type);
        }
        println!("");
        std::process::exit(0);
    }
}

pub fn main_loop<T: ExampleApp>(mut app: T) {
    info!("Starting main loop...");
    let mut stats = FrameStats::default();

    'running: loop {
        let events = app.poll_events();
        for event in events {
            match event {
                i3_gfx::graph::backend::Event::Quit
                | i3_gfx::graph::backend::Event::KeyDown {
                    key: i3_gfx::graph::backend::KeyCode::Escape,
                } => break 'running,
                _ => {}
            }
            app.handle_event(&event);
        }

        let delta = stats.update();
        app.update(delta, stats.smoothed_dt());
        app.render();
    }
    info!("Main loop finished. Shutting down...");
}
