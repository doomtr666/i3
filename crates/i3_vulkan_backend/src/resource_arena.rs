//! # Resource Arena - Generational Index System
//!
//! This module implements a resource management system based on **generational indices**.
//! This is a critical pattern for memory safety in a rendering engine.

use ash::vk;
use i3_gfx::graph::types::*;
use i3_gfx::graph::backend::BlasCreateInfo;

/// Physical representation of a Vulkan pipeline.
#[derive(Clone)]
pub struct PhysicalPipeline {
    pub handle: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub bind_point: vk::PipelineBindPoint,
    pub set_layouts: Vec<vk::DescriptorSetLayout>,
    pub physical_id: u64,
}

/// Physical representation of a Vulkan image.
pub struct PhysicalImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub allocation: Option<vk_mem::Allocation>,
    pub desc: ImageDesc,
    pub format: vk::Format,

    pub last_layout: vk::ImageLayout,
    pub last_access: vk::AccessFlags2,
    pub last_stage: vk::PipelineStageFlags2,
    pub last_write_frame: u64,
    pub last_queue_family: u32,
    pub is_swapchain: bool,
    pub concurrent: bool,
    pub is_transient: bool,

    pub subresource_views: std::sync::Mutex<std::collections::HashMap<(u32, u32), vk::ImageView>>,
}

/// Physical representation of a Vulkan buffer.
pub struct PhysicalBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<vk_mem::Allocation>,
    pub desc: BufferDesc,

    pub last_access: vk::AccessFlags2,
    pub last_stage: vk::PipelineStageFlags2,
    pub last_queue_family: u32,
    pub concurrent: bool,
    pub is_transient: bool,
}

/// Physical representation of a Vulkan Acceleration Structure (TLAS/BLAS).
pub struct PhysicalAccelerationStructure {
    pub handle: vk::AccelerationStructureKHR,
    pub buffer: vk::Buffer,
    pub allocation: Option<vk_mem::Allocation>,
    pub desc: AccelerationStructureDesc,
    pub is_transient: bool,
    pub build_info: Option<BlasCreateInfo>,
    pub address: u64,
}

/// Slot in the arena - either occupied with data or free with generation tracking.
enum Slot<T> {
    Occupied {
        data: T,
        generation: u32,
    },
    Free {
        next_free: Option<u32>,
        generation: u32,
    },
}

/// Generic resource arena with generational indices.
pub struct ResourceArena<T> {
    slots: Vec<Slot<T>>,
    free_head: Option<u32>,
}

impl<T> ResourceArena<T> {
    pub fn new() -> Self {
        Self {
            slots: Vec::with_capacity(256),
            free_head: None,
        }
    }

    pub fn insert(&mut self, data: T) -> u64 {
        if let Some(index) = self.free_head {
            let slot = &mut self.slots[index as usize];
            if let Slot::Free {
                next_free,
                generation,
            } = *slot
            {
                let generation_val = generation;
                *slot = Slot::Occupied {
                    data,
                    generation: generation_val,
                };
                self.free_head = next_free;
                return ((generation_val as u64) << 32) | (index as u64);
            }
        }

        let index = self.slots.len() as u32;
        let generation = 1u32;
        self.slots.push(Slot::Occupied { data, generation });
        ((generation as u64) << 32) | (index as u64)
    }

    pub fn get(&self, id: u64) -> Option<&T> {
        let index = (id & 0xFFFFFFFF) as usize;
        let generation_val = (id >> 32) as u32;
        if let Some(Slot::Occupied { data, generation }) = self.slots.get(index) {
            if *generation == generation_val {
                return Some(data);
            }
        }
        None
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut T> {
        let index = (id & 0xFFFFFFFF) as usize;
        let generation_val = (id >> 32) as u32;
        if let Some(Slot::Occupied { data, generation }) = self.slots.get_mut(index) {
            if *generation == generation_val {
                return Some(data);
            }
        }
        None
    }

    pub fn remove(&mut self, id: u64) -> Option<T> {
        let index = (id & 0xFFFFFFFF) as usize;
        let generation_val = (id >> 32) as u32;
        if index >= self.slots.len() {
            return None;
        }

        match self.slots[index] {
            Slot::Occupied { generation, .. } if generation == generation_val => {
                let old_slot = std::mem::replace(
                    &mut self.slots[index],
                    Slot::Free {
                        next_free: self.free_head,
                        generation: generation + 1,
                    },
                );
                self.free_head = Some(index as u32);
                if let Slot::Occupied { data, .. } = old_slot {
                    Some(data)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u64, &mut T)> {
        self.slots.iter_mut().enumerate().filter_map(|(i, slot)| {
            if let Slot::Occupied { data, generation } = slot {
                let id = ((*generation as u64) << 32) | (i as u64);
                Some((id, data))
            } else {
                None
            }
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (u64, &T)> {
        self.slots.iter().enumerate().filter_map(|(i, slot)| {
            if let Slot::Occupied { data, generation } = slot {
                let id = ((*generation as u64) << 32) | (i as u64);
                Some((id, data))
            } else {
                None
            }
        })
    }

    pub fn ids(&self) -> Vec<u64> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| {
                if let Slot::Occupied { generation, .. } = slot {
                    Some(((*generation as u64) << 32) | (i as u64))
                } else {
                    None
                }
            })
            .collect()
    }
}
