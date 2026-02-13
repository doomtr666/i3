use i3_gfx::graph::hri::{HriBackend, PassContext};
use log::info;

pub struct NullBackend;

impl HriBackend for NullBackend {
    // Implementations will log resource creation, submission, etc.
}

pub struct NullPassContext {
    pass_name: String,
}

impl NullPassContext {
    pub fn new(name: &str) -> Self {
        Self {
            pass_name: name.to_string(),
        }
    }
}

impl PassContext for NullPassContext {
    fn draw(&mut self, vertex_count: u32, first_vertex: u32) {
        info!(
            "[{}] Draw: count={}, first={}",
            self.pass_name, vertex_count, first_vertex
        );
    }

    fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        info!("[{}] Dispatch: [{}, {}, {}]", self.pass_name, x, y, z);
    }
}
