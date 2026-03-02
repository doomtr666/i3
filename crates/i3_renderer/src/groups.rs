use crate::passes::average_luminance::AverageLuminancePass;
use crate::passes::cluster_build::{ClusterBuildPass, ClusterBuildPushConstants};
use crate::passes::histogram_build::HistogramBuildPass;
use crate::passes::light_cull::{LightCullPass, LightCullPushConstants};
use crate::passes::tonemap::TonemapPass;
use i3_gfx::prelude::*;
use std::sync::{Arc, Mutex};

pub struct ClusteringGroup {
    pub cluster_build_pass: Arc<Mutex<ClusterBuildPass>>,
    pub light_cull_pass: Arc<Mutex<LightCullPass>>,
}

impl ClusteringGroup {
    pub fn new(
        cluster_aabbs: BufferHandle,
        lights: BufferHandle,
        cluster_grid: BufferHandle,
        cluster_light_indices: BufferHandle,
        grid_size: [u32; 3],
    ) -> Self {
        Self {
            cluster_build_pass: Arc::new(Mutex::new(ClusterBuildPass::new(
                cluster_aabbs,
                ClusterBuildPushConstants {
                    inv_projection: nalgebra_glm::identity(),
                    grid_size,
                    near_plane: 0.0,
                    far_plane: 0.0,
                    screen_dimensions: [0.0, 0.0],
                    pad: 0,
                },
            ))),
            light_cull_pass: Arc::new(Mutex::new(LightCullPass::new(
                cluster_aabbs,
                lights,
                cluster_grid,
                cluster_light_indices,
                LightCullPushConstants {
                    view_matrix: nalgebra_glm::identity(),
                    grid_size,
                    light_count: 0,
                },
            ))),
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
    pub fn new(
        hdr_target: ImageHandle,
        backbuffer: ImageHandle,
        histogram_buffer: BufferHandle,
        exposure_buffer: BufferHandle,
        sampler: SamplerHandle,
    ) -> Self {
        Self {
            histogram_build_pass: Arc::new(Mutex::new(HistogramBuildPass::new(
                hdr_target,
                histogram_buffer,
                exposure_buffer,
            ))),
            average_luminance_pass: Arc::new(Mutex::new(AverageLuminancePass::new(
                histogram_buffer,
                exposure_buffer,
            ))),
            tonemap_pass: Arc::new(Mutex::new(TonemapPass::new(
                backbuffer,
                hdr_target,
                exposure_buffer,
                sampler,
            ))),
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
