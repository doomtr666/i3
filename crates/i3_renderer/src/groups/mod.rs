use crate::passes::average_luminance::AverageLuminancePass;
use crate::passes::cluster_build::ClusterBuildPass;
use crate::passes::histogram_build::HistogramBuildPass;
use crate::passes::light_cull::LightCullPass;
use crate::passes::tonemap::TonemapPass;
use i3_gfx::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// ClusteringGroup — declares cluster buffers as outputs, adds clear + children
// ─────────────────────────────────────────────────────────────────────────────

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

    fn declare(&mut self, builder: &mut PassBuilder) {
        let common = *builder.consume::<crate::render_graph::CommonData>("Common");
        let grid_x = (common.screen_width + crate::constants::CLUSTER_TILE_SIZE - 1) / crate::constants::CLUSTER_TILE_SIZE;
        let grid_y = (common.screen_height + crate::constants::CLUSTER_TILE_SIZE - 1) / crate::constants::CLUSTER_TILE_SIZE;
        let grid_z: u32 = crate::constants::CLUSTER_GRID_Z;
        let max_clusters = (grid_x * grid_y * grid_z) as u64;

        builder.declare_buffer_output("ClusterAABBs", BufferDesc {
            size: max_clusters * 32,
            usage: BufferUsageFlags::STORAGE_BUFFER,
            memory: MemoryType::GpuOnly,
        });
        builder.declare_buffer_output("ClusterGrid", BufferDesc {
            size: max_clusters * 8,
            usage: BufferUsageFlags::STORAGE_BUFFER,
            memory: MemoryType::GpuOnly,
        });
        let cluster_light_indices = builder.declare_buffer_output("ClusterLightIndices", BufferDesc {
            size: max_clusters * 256 * 4,
            usage: BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
            memory: MemoryType::GpuOnly,
        });

        builder.publish("ClusterGridSize", [grid_x, grid_y, grid_z]);

        builder.add_owned_pass(crate::render_graph::ClearBufferPass {
            name: "ClearClusterIndices".to_string(),
            buffer: cluster_light_indices,
        });
        builder.add_pass(&mut self.cluster_build_pass);
        builder.add_pass(&mut self.light_cull_pass);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PostProcessGroup — declares HistogramBuffer as output, adds clear + children
// ─────────────────────────────────────────────────────────────────────────────

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

    fn declare(&mut self, builder: &mut PassBuilder) {
        let histogram_buffer = builder.declare_buffer_output("HistogramBuffer", BufferDesc {
            size: 256 * 4,
            usage: BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
            memory: MemoryType::GpuOnly,
        });

        builder.add_owned_pass(crate::render_graph::ClearBufferPass {
            name: "ClearHistogram".to_string(),
            buffer: histogram_buffer,
        });
        builder.add_pass(&mut self.histogram_build_pass);
        builder.add_pass(&mut self.average_luminance_pass);
        builder.add_pass(&mut self.tonemap_pass);
    }
}

pub mod sync;
