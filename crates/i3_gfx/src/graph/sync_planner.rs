use std::collections::HashMap;
use crate::graph::types::{FlatPass, ResourceUsage, ImageHandle, BufferHandle, PassDomain};
use crate::graph::sync::{SyncPlan, ResourceState, PassSyncData, AbstractTransition, ResourceKind, TransitionKind, LoadOp, ImageLayout, AccessFlags, StageFlags};

#[derive(Debug, Clone, Copy)]
pub struct ResourceFlowState {
    pub state: ResourceState,
    pub last_write_pass_idx: Option<usize>,
}

pub struct SyncPlanner {
    pub image_flow: HashMap<u64, ResourceFlowState>,
    pub buffer_flow: HashMap<u64, ResourceFlowState>,
    pub as_flow: HashMap<u64, ResourceFlowState>,
    // Scratch maps for seeding — reused across frames, owned here to avoid per-frame alloc.
    // Access via seed_image / seed_buffer / clear_seeds.
    image_seed: HashMap<u64, ResourceState>,
    buffer_seed: HashMap<u64, ResourceState>,
    // Scratch buffers reused across frames — grow to high-water mark then stop allocating.
    scratch_images: Vec<(ImageHandle, ResourceUsage)>,
    scratch_buffers: Vec<(BufferHandle, ResourceUsage)>,
}

impl SyncPlanner {
    pub fn new() -> Self {
        Self {
            image_flow: HashMap::new(),
            buffer_flow: HashMap::new(),
            as_flow: HashMap::new(),
            image_seed: HashMap::new(),
            buffer_seed: HashMap::new(),
            scratch_images: Vec::with_capacity(8),
            scratch_buffers: Vec::with_capacity(16),
        }
    }

    /// Register the last-known state of a virtual image before this frame's analysis.
    /// Called by the backend after translating physical resource state to abstract form.
    pub fn seed_image(&mut self, virtual_id: u64, state: ResourceState) {
        self.image_seed.insert(virtual_id, state);
    }

    /// Register the last-known state of a virtual buffer before this frame's analysis.
    pub fn seed_buffer(&mut self, virtual_id: u64, state: ResourceState) {
        self.buffer_seed.insert(virtual_id, state);
    }

    /// Clear all seed state (call at the start of each frame before re-seeding).
    pub fn clear_seeds(&mut self) {
        self.image_seed.clear();
        self.buffer_seed.clear();
    }

    pub fn analyze(&mut self, passes: &[FlatPass]) -> SyncPlan {
        let total_resources = self.image_seed.len() + self.buffer_seed.len();
        let mut plan = SyncPlan {
            passes: vec![PassSyncData::default(); passes.len()],
            final_states: HashMap::with_capacity(total_resources),
        };

        // 1. Seed initial states from the caller-populated seed maps.
        self.image_flow.clear();
        self.buffer_flow.clear();
        self.as_flow.clear();
        for (&id, &state) in &self.image_seed {
            self.image_flow.insert(id, ResourceFlowState { state, last_write_pass_idx: None });
        }
        for (&id, &state) in &self.buffer_seed {
            self.buffer_flow.insert(id, ResourceFlowState { state, last_write_pass_idx: None });
        }

        // 2. Simulation Loop
        for (idx, pass) in passes.iter().enumerate() {
            self.simulate_pass(pass, idx, &mut plan);
        }

        // 3. Export final states
        for (&id, flow) in &self.image_flow {
            plan.final_states.insert(id, flow.state);
        }
        for (&id, flow) in &self.buffer_flow {
            plan.final_states.insert(id, flow.state);
        }

        plan
    }

    fn simulate_pass(&mut self, pass: &FlatPass, pass_idx: usize, plan: &mut SyncPlan) {
        let current_queue = pass.queue;

        let bind_point = match pass.domain {
            PassDomain::Graphics => BindingPoint::Graphics,
            _ => BindingPoint::Compute,
        };

        // --- Process Images ---
        // Linear dedup into a scratch Vec — passes touch < 8 images, O(n²) beats HashMap here.
        // PRESENT is excluded: it is declared in image_reads only for DAG dependency ordering;
        // the actual layout transition is handled separately in the post_transitions block below.
        self.scratch_images.clear();
        for (handle, usage) in pass.image_reads.iter().chain(pass.image_writes.iter()) {
            let usage = usage.difference(ResourceUsage::PRESENT);
            if usage.is_empty() { continue; }
            if let Some(existing) = self.scratch_images.iter_mut().find(|(h, _)| h == handle) {
                existing.1 |= usage;
            } else {
                self.scratch_images.push((*handle, usage));
            }
        }

        for &(handle, usage) in &self.scratch_images {
            let id = handle.0.0;
            let old_flow = self.image_flow.get(&id).cloned().unwrap_or(ResourceFlowState {
                state: ResourceState {
                    layout: ImageLayout::Undefined,
                    access: AccessFlags::NONE,
                    stage: StageFlags::TOP_OF_PIPE,
                    queue: crate::graph::types::QueueType::Graphics,
                },
                last_write_pass_idx: None,
            });

            let old_state = old_flow.state;
            let (target_layout, target_access, target_stage) = get_image_state(usage, bind_point);

            let new_state = ResourceState {
                layout: target_layout,
                access: target_access,
                stage: target_stage,
                queue: current_queue,
            };

            // Layout Promotion (First Use)
            if old_state.layout == ImageLayout::Undefined {
                if usage.intersects(ResourceUsage::COLOR_ATTACHMENT | ResourceUsage::DEPTH_STENCIL | ResourceUsage::CLEAR) {
                    plan.passes[pass_idx].load_ops.insert(id, LoadOp::Clear);
                }
            }

            // Barrier Generation
            if needs_barrier(&old_state, &new_state) {
                // Cross-queue: normalize old_state stage/access to queue-agnostic values so
                // the barrier is valid regardless of which queue records it.
                // MEMORY_READ|WRITE with ALL_COMMANDS = "flush/invalidate all caches" — conservative
                // but always valid. Proper Release/Acquire pairs (SYNC-01) would be more precise.
                let effective_old = if old_state.queue != current_queue {
                    ResourceState {
                        layout: old_state.layout,
                        access: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                        stage: StageFlags::ALL_COMMANDS,
                        queue: old_state.queue,
                    }
                } else {
                    old_state
                };
                plan.passes[pass_idx].pre_transitions.push(AbstractTransition {
                    resource_id: id,
                    resource_kind: ResourceKind::Image,
                    old_state: effective_old,
                    new_state,
                    kind: TransitionKind::Regular,
                });
            }

            // Update flow state
            let flow = self.image_flow.entry(id).or_insert(ResourceFlowState {
                state: new_state,
                last_write_pass_idx: None,
            });
            flow.state = new_state;
            if usage.intersects(ResourceUsage::WRITE | ResourceUsage::COLOR_ATTACHMENT | ResourceUsage::DEPTH_STENCIL) {
                flow.last_write_pass_idx = Some(pass_idx);
            }
        }

        // --- Process Buffers ---
        self.scratch_buffers.clear();
        for (handle, usage) in pass.buffer_reads.iter().chain(pass.buffer_writes.iter()) {
            if let Some(existing) = self.scratch_buffers.iter_mut().find(|(h, _)| h == handle) {
                existing.1 |= *usage;
            } else {
                self.scratch_buffers.push((*handle, *usage));
            }
        }

        for &(handle, usage) in &self.scratch_buffers {
            let id = handle.0.0;
            let old_flow = self.buffer_flow.get(&id).cloned().unwrap_or(ResourceFlowState {
                state: ResourceState::default(),
                last_write_pass_idx: None,
            });
            let old_state = old_flow.state;
            let (target_access, target_stage) = get_buffer_state(usage, bind_point);
            
            let new_state = ResourceState {
                layout: ImageLayout::Undefined,
                access: target_access,
                stage: target_stage,
                queue: current_queue,
            };

            if needs_barrier(&old_state, &new_state) {
                let effective_old = if old_state.queue != current_queue {
                    ResourceState {
                        layout: ImageLayout::Undefined,
                        access: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                        stage: StageFlags::ALL_COMMANDS,
                        queue: old_state.queue,
                    }
                } else {
                    old_state
                };
                plan.passes[pass_idx].pre_transitions.push(AbstractTransition {
                    resource_id: id,
                    resource_kind: ResourceKind::Buffer,
                    old_state: effective_old,
                    new_state,
                    kind: TransitionKind::Regular,
                });
            }

            let flow = self.buffer_flow.entry(id).or_insert(ResourceFlowState {
                state: new_state,
                last_write_pass_idx: None,
            });
            flow.state = new_state;
            if usage.intersects(ResourceUsage::WRITE) {
                flow.last_write_pass_idx = Some(pass_idx);
            }
        }

        // --- Process Acceleration Structures ---
        // AS use global VkMemoryBarrier2 (no per-resource layout, accessed via device address).
        // We reuse get_buffer_state for access/stage mapping.
        for &(handle, usage) in pass.accel_struct_reads.iter().chain(pass.accel_struct_writes.iter()) {
            let id = handle.0.0;
            let old_flow = self.as_flow.get(&id).cloned().unwrap_or(ResourceFlowState {
                state: ResourceState::default(),
                last_write_pass_idx: None,
            });
            let old_state = old_flow.state;
            let (target_access, target_stage) = get_buffer_state(usage, bind_point);
            let new_state = ResourceState {
                layout: ImageLayout::Undefined,
                access: target_access,
                stage: target_stage,
                queue: current_queue,
            };

            if needs_barrier(&old_state, &new_state) {
                let effective_old = if old_state.queue != current_queue {
                    ResourceState {
                        layout: ImageLayout::Undefined,
                        access: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                        stage: StageFlags::ALL_COMMANDS,
                        queue: old_state.queue,
                    }
                } else {
                    old_state
                };
                plan.passes[pass_idx].pre_transitions.push(AbstractTransition {
                    resource_id: id,
                    resource_kind: ResourceKind::AccelStruct,
                    old_state: effective_old,
                    new_state,
                    kind: TransitionKind::Regular,
                });
            }

            let flow = self.as_flow.entry(id).or_insert(ResourceFlowState {
                state: new_state,
                last_write_pass_idx: None,
            });
            flow.state = new_state;
            if usage.intersects(ResourceUsage::ACCEL_STRUCT_WRITE) {
                flow.last_write_pass_idx = Some(pass_idx);
            }
        }

        // --- Process Present Images (post-transitions) ---
        // These run AFTER the pass executes: transition from whatever state the image
        // ended up in (post-write) to PresentSrc/BOTTOM_OF_PIPE.
        for &handle in &pass.present_images {
            let id = handle.0.0;
            let old_flow = self.image_flow.get(&id).cloned().unwrap_or(ResourceFlowState {
                state: ResourceState {
                    layout: ImageLayout::Undefined,
                    access: AccessFlags::NONE,
                    stage: StageFlags::TOP_OF_PIPE,
                    queue: current_queue,
                },
                last_write_pass_idx: None,
            });
            let old_state = old_flow.state;
            let present_state = ResourceState {
                layout: ImageLayout::PresentSrc,
                access: AccessFlags::NONE,
                stage: StageFlags::BOTTOM_OF_PIPE,
                queue: current_queue,
            };

            // The barrier source must synchronize with the prior write on this queue.
            // Cross-queue normalization applies here too if the image was previously
            // on a different queue.
            let effective_old = if old_state.queue != current_queue {
                ResourceState {
                    layout: old_state.layout,
                    access: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                    stage: StageFlags::ALL_COMMANDS,
                    queue: old_state.queue,
                }
            } else {
                old_state
            };

            plan.passes[pass_idx].post_transitions.push(AbstractTransition {
                resource_id: id,
                resource_kind: ResourceKind::Image,
                old_state: effective_old,
                new_state: present_state,
                kind: TransitionKind::Regular,
            });

            // Update flow so subsequent passes see PresentSrc as the starting state.
            let flow = self.image_flow.entry(id).or_insert(ResourceFlowState {
                state: present_state,
                last_write_pass_idx: None,
            });
            flow.state = present_state;
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum BindingPoint {
    Graphics,
    Compute,
}

const WRITE_ACCESSES: AccessFlags = AccessFlags::SHADER_WRITE
    .union(AccessFlags::COLOR_ATTACHMENT_WRITE)
    .union(AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
    .union(AccessFlags::TRANSFER_WRITE)
    .union(AccessFlags::HOST_WRITE)
    .union(AccessFlags::MEMORY_WRITE)
    .union(AccessFlags::ACCELERATION_STRUCTURE_WRITE);

fn needs_barrier(old: &ResourceState, new: &ResourceState) -> bool {
    // Layout transition always requires a barrier.
    old.layout != new.layout
    // Any previous write must be made visible/available before any subsequent access (RAW, WAW).
    || old.access.intersects(WRITE_ACCESSES)
    // Any new write requires an execution dependency on prior accesses (WAR, WAW).
    || new.access.intersects(WRITE_ACCESSES)
    // Queue ownership transfer.
    || old.queue != new.queue
}

fn get_image_state(usage: ResourceUsage, bind_point: BindingPoint) -> (ImageLayout, AccessFlags, StageFlags) {
    // Priority order: attachment > transfer > shader > present
    // CLEAR on an image = vkCmdClearColorImage or renderpass loadOp=CLEAR (handled separately via load_ops);
    // when declared as a standalone CLEAR (no attachment flag), use transfer dst.
    if usage.contains(ResourceUsage::COLOR_ATTACHMENT) {
        return (ImageLayout::ColorAttachmentOptimal, AccessFlags::COLOR_ATTACHMENT_WRITE | AccessFlags::COLOR_ATTACHMENT_READ, StageFlags::COLOR_ATTACHMENT_OUTPUT);
    }
    if usage.contains(ResourceUsage::DEPTH_STENCIL) {
        return (ImageLayout::DepthStencilAttachmentOptimal, AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE | AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ, StageFlags::EARLY_FRAGMENT_TESTS | StageFlags::LATE_FRAGMENT_TESTS);
    }
    if usage.contains(ResourceUsage::TRANSFER_WRITE) || usage.contains(ResourceUsage::CLEAR) {
        return (ImageLayout::TransferDstOptimal, AccessFlags::TRANSFER_WRITE, StageFlags::TRANSFER);
    }
    if usage.contains(ResourceUsage::TRANSFER_READ) {
        return (ImageLayout::TransferSrcOptimal, AccessFlags::TRANSFER_READ, StageFlags::TRANSFER);
    }
    if usage.contains(ResourceUsage::SHADER_WRITE) {
        return (ImageLayout::General, AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE, match bind_point {
            BindingPoint::Compute => StageFlags::COMPUTE_SHADER,
            BindingPoint::Graphics => StageFlags::ALL_GRAPHICS,
        });
    }
    if usage.contains(ResourceUsage::SHADER_READ) {
        return (ImageLayout::ShaderReadOnlyOptimal, AccessFlags::SHADER_READ, match bind_point {
            BindingPoint::Compute => StageFlags::COMPUTE_SHADER,
            BindingPoint::Graphics => StageFlags::ALL_GRAPHICS,
        });
    }
    if usage.contains(ResourceUsage::PRESENT) {
        return (ImageLayout::PresentSrc, AccessFlags::NONE, StageFlags::BOTTOM_OF_PIPE);
    }
    (ImageLayout::Undefined, AccessFlags::NONE, StageFlags::TOP_OF_PIPE)
}

fn get_buffer_state(usage: ResourceUsage, bind_point: BindingPoint) -> (AccessFlags, StageFlags) {
    let mut access = AccessFlags::NONE;
    let mut stage = StageFlags::empty();

    if usage.contains(ResourceUsage::SHADER_READ) {
        access |= AccessFlags::SHADER_READ;
        stage |= match bind_point {
            BindingPoint::Compute => StageFlags::COMPUTE_SHADER,
            BindingPoint::Graphics => StageFlags::ALL_GRAPHICS,
        };
    }
    if usage.contains(ResourceUsage::SHADER_WRITE) {
        access |= AccessFlags::SHADER_WRITE;
        stage |= match bind_point {
            BindingPoint::Compute => StageFlags::COMPUTE_SHADER,
            BindingPoint::Graphics => StageFlags::ALL_GRAPHICS,
        };
    }
    if usage.contains(ResourceUsage::TRANSFER_READ) {
        access |= AccessFlags::TRANSFER_READ;
        stage |= StageFlags::TRANSFER;
    }
    if usage.contains(ResourceUsage::TRANSFER_WRITE) {
        access |= AccessFlags::TRANSFER_WRITE;
        stage |= StageFlags::TRANSFER;
    }
    // vkCmdFillBuffer uses the CLEAR stage (not TRANSFER) in sync2
    if usage.contains(ResourceUsage::CLEAR) {
        access |= AccessFlags::TRANSFER_WRITE;
        stage |= StageFlags::CLEAR;
    }
    if usage.contains(ResourceUsage::INDIRECT_READ) {
        access |= AccessFlags::INDIRECT_COMMAND_READ;
        stage |= StageFlags::DRAW_INDIRECT;
    }
    if usage.contains(ResourceUsage::ACCEL_STRUCT_READ) {
        access |= AccessFlags::ACCELERATION_STRUCTURE_READ;
        // Cover both AS build-time reads and ray tracing shader reads.
        stage |= StageFlags::ACCELERATION_STRUCTURE_BUILD | StageFlags::RAY_TRACING_SHADER;
    }
    if usage.contains(ResourceUsage::ACCEL_STRUCT_WRITE) {
        access |= AccessFlags::ACCELERATION_STRUCTURE_WRITE;
        stage |= StageFlags::ACCELERATION_STRUCTURE_BUILD;
    }

    if stage.is_empty() {
        stage = StageFlags::TOP_OF_PIPE;
    }

    (access, stage)
}
