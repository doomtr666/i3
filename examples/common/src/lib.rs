use std::time::{Duration, Instant};
use tracing::info;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

pub fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,i3_vulkan_backend=info,i3_gfx=info"));

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();
}

pub struct FrameStats {
    last_frame: Instant,
    frame_count: u32,
    accumulated_time: Duration,
}

impl Default for FrameStats {
    fn default() -> Self {
        Self {
            last_frame: Instant::now(),
            frame_count: 0,
            accumulated_time: Duration::ZERO,
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

        if self.frame_count >= 1000 {
            let avg = self.accumulated_time.as_secs_f32() / self.frame_count as f32;
            info!(
                "Frame Stats: Avg Frame Time: {:.2}ms ({:.1} FPS)",
                avg * 1000.0,
                1.0 / avg
            );
            self.frame_count = 0;
            self.accumulated_time = Duration::ZERO;
        }

        delta
    }
}

pub trait ExampleApp {
    fn update(&mut self, delta: Duration);
    fn render(&mut self);
    fn poll_events(&mut self) -> Vec<i3_gfx::graph::backend::Event>;
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
        }

        let delta = stats.update();
        app.update(delta);
        app.render();
    }
}
