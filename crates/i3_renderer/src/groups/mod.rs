use crate::passes::average_luminance::AverageLuminancePass;
use crate::passes::cluster_build::ClusterBuildPass;
use crate::passes::histogram_build::HistogramBuildPass;
use crate::passes::light_cull::LightCullPass;
use crate::passes::tonemap::TonemapPass;
use i3_gfx::prelude::*;
use std::sync::{Arc, Mutex};

pub struct ClusteringGroup {
    pub cluster_build_pass: Arc<Mutex<ClusterBuildPass>>,
    pub light_cull_pass: Arc<Mutex<LightCullPass>>,
}

impl ClusteringGroup {
    pub fn new() -> Self {
        Self {
            cluster_build_pass: Arc::new(Mutex::new(ClusterBuildPass::new())),
            light_cull_pass: Arc::new(Mutex::new(LightCullPass::new())),
        }
    }
}

impl RenderPass for ClusteringGroup {
    fn name(&self) -> &str {
        "ClusteringGroup"
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        builder.add_pass(self.cluster_build_pass.clone());
        builder.add_pass(self.light_cull_pass.clone());
    }
}

pub struct PostProcessGroup {
    pub histogram_build_pass: Arc<Mutex<HistogramBuildPass>>,
    pub average_luminance_pass: Arc<Mutex<AverageLuminancePass>>,
    pub tonemap_pass: Arc<Mutex<TonemapPass>>,
}

impl PostProcessGroup {
    pub fn new(sampler: SamplerHandle) -> Self {
        Self {
            histogram_build_pass: Arc::new(Mutex::new(HistogramBuildPass::new())),
            average_luminance_pass: Arc::new(Mutex::new(AverageLuminancePass::new())),
            tonemap_pass: Arc::new(Mutex::new(TonemapPass::new(sampler))),
        }
    }
}

impl RenderPass for PostProcessGroup {
    fn name(&self) -> &str {
        "PostProcessGroup"
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        builder.add_pass(self.histogram_build_pass.clone());
        builder.add_pass(self.average_luminance_pass.clone());
        builder.add_pass(self.tonemap_pass.clone());
    }
}

pub mod sync;
