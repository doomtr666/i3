use crate::passes::average_luminance::AverageLuminancePass;
use crate::passes::cluster_build::ClusterBuildPass;
use crate::passes::histogram_build::HistogramBuildPass;
use crate::passes::light_cull::LightCullPass;
use crate::passes::tonemap::TonemapPass;
use i3_gfx::prelude::*;
pub struct ClusteringGroup {
    pub cluster_build_pass: ClusterBuildPass,
    pub light_cull_pass: LightCullPass,
}

impl ClusteringGroup {
    pub fn new() -> Self {
        Self {
            cluster_build_pass: ClusterBuildPass::new(),
            light_cull_pass: LightCullPass::new(),
        }
    }
}

impl RenderPass for ClusteringGroup {
    fn name(&self) -> &str {
        "ClusteringGroup"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        self.cluster_build_pass.init(backend, globals);
        self.light_cull_pass.init(backend, globals);
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        builder.add_pass(&mut self.cluster_build_pass);
        builder.add_pass(&mut self.light_cull_pass);
    }
}

pub struct PostProcessGroup {
    pub histogram_build_pass: HistogramBuildPass,
    pub average_luminance_pass: AverageLuminancePass,
    pub tonemap_pass: TonemapPass,
}

impl PostProcessGroup {
    pub fn new(sampler: SamplerHandle) -> Self {
        Self {
            histogram_build_pass: HistogramBuildPass::new(),
            average_luminance_pass: AverageLuminancePass::new(),
            tonemap_pass: TonemapPass::new(sampler),
        }
    }
}

impl RenderPass for PostProcessGroup {
    fn name(&self) -> &str {
        "PostProcessGroup"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        self.histogram_build_pass.init(backend, globals);
        self.average_luminance_pass.init(backend, globals);
        self.tonemap_pass.init(backend, globals);
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        builder.add_pass(&mut self.histogram_build_pass);
        builder.add_pass(&mut self.average_luminance_pass);
        builder.add_pass(&mut self.tonemap_pass);
    }
}

pub mod sync;
