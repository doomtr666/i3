use crate::passes::rtao::RtaoPass;
use i3_gfx::prelude::*;

pub struct RtaoGroup {
    pub rtao_pass: RtaoPass,
}

impl RtaoGroup {
    pub fn new() -> Self {
        Self { rtao_pass: RtaoPass::new() }
    }

    pub fn tick(&mut self) {
        self.rtao_pass.tick();
    }
}

impl RenderPass for RtaoGroup {
    fn name(&self) -> &str {
        "RtaoGroup"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        self.rtao_pass.init(backend, globals);
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        builder.add_pass(&mut self.rtao_pass);
    }
}
