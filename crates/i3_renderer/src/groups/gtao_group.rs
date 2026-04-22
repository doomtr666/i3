use crate::passes::gtao::GtaoPass;
use crate::passes::gtao_temporal::GtaoTemporalPass;
use i3_gfx::prelude::*;

pub struct GtaoGroup {
    pub gtao_pass: GtaoPass,
    pub gtao_temporal_pass: GtaoTemporalPass,
}

impl GtaoGroup {
    pub fn new() -> Self {
        Self {
            gtao_pass: GtaoPass::new(),
            gtao_temporal_pass: GtaoTemporalPass::new(),
        }
    }

    pub fn tick(&mut self) {
        self.gtao_pass.tick();
        self.gtao_temporal_pass.tick();
    }
}

impl RenderPass for GtaoGroup {
    fn name(&self) -> &str {
        "GtaoGroup"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        self.gtao_pass.init(backend, globals);
        self.gtao_temporal_pass.init(backend, globals);
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        builder.add_pass(&mut self.gtao_pass);
        builder.add_pass(&mut self.gtao_temporal_pass);
    }
}
