use crate::graph::backend::{BackendBuffer, BackendImage, RenderBackendInternal};
use crate::graph::types::{BufferDesc, ImageDesc};
use std::collections::HashMap;

/// Registry that persists across frames to manage temporal history resources.
/// It holds double-buffered physical Vulkan resources based on their graph name.
pub struct TemporalRegistry {
    // Current logical frame index (0..capacity-1).
    pub current_frame: usize,
    // Maximum number of frames in flight (usually matches swapchain image count).
    pub capacity: usize,

    pub(crate) buffers: HashMap<String, Vec<BackendBuffer>>,
    pub(crate) images: HashMap<String, Vec<BackendImage>>,
}

impl Default for TemporalRegistry {
    fn default() -> Self {
        Self {
            current_frame: 0,
            capacity: 2, // Default to double buffering
            buffers: HashMap::new(),
            images: HashMap::new(),
        }
    }
}

impl TemporalRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Called at the beginning of a frame to flip history buffers.
    /// capacity should match the number of frames in flight (e.g. swapchain count).
    pub fn advance_frame(&mut self, capacity: usize) {
        self.capacity = capacity.max(2);
        self.current_frame = (self.current_frame + 1) % self.capacity;
    }

    /// Retrieve or create a persistent N-buffered physical buffer.
    pub fn get_or_create_buffer<B: RenderBackendInternal>(
        &mut self,
        name: &str,
        desc: &BufferDesc,
        backend: &mut B,
    ) -> BackendBuffer {
        let cap = self.capacity;
        let entry = self.buffers.entry(name.to_string()).or_insert_with(|| {
            (0..cap)
                .map(|_| backend.create_transient_buffer(desc))
                .collect()
        });
        
        // If capacity increased, grow the pool
        if entry.len() < cap {
            for _ in entry.len()..cap {
                entry.push(backend.create_transient_buffer(desc));
            }
        }

        entry[self.current_frame % entry.len()].clone()
    }

    /// Retrieve the history buffer (N-1) for a persistent N-buffered physical buffer.
    pub fn get_or_create_history_buffer<B: RenderBackendInternal>(
        &mut self,
        name: &str,
        desc: &BufferDesc,
        backend: &mut B,
    ) -> BackendBuffer {
        let cap = self.capacity;
        let entry = self.buffers.entry(name.to_string()).or_insert_with(|| {
            (0..cap)
                .map(|_| backend.create_transient_buffer(desc))
                .collect()
        });
        
        if entry.len() < cap {
            for _ in entry.len()..cap {
                entry.push(backend.create_transient_buffer(desc));
            }
        }

        let history_idx = (self.current_frame + entry.len() - 1) % entry.len();
        entry[history_idx].clone()
    }

    /// Retrieve or create a persistent N-buffered physical image.
    pub fn get_or_create_image<B: RenderBackendInternal>(
        &mut self,
        name: &str,
        desc: &ImageDesc,
        backend: &mut B,
    ) -> BackendImage {
        let cap = self.capacity;
        let entry = self.images.entry(name.to_string()).or_insert_with(|| {
            (0..cap)
                .map(|_| backend.create_transient_image(desc))
                .collect()
        });

        if entry.len() < cap {
            for _ in entry.len()..cap {
                entry.push(backend.create_transient_image(desc));
            }
        }

        entry[self.current_frame % entry.len()]
    }

    /// Retrieve the history image (N-1).
    pub fn get_or_create_history_image<B: RenderBackendInternal>(
        &mut self,
        name: &str,
        desc: &ImageDesc,
        backend: &mut B,
    ) -> BackendImage {
        let cap = self.capacity;
        let entry = self.images.entry(name.to_string()).or_insert_with(|| {
            (0..cap)
                .map(|_| backend.create_transient_image(desc))
                .collect()
        });

        if entry.len() < cap {
            for _ in entry.len()..cap {
                entry.push(backend.create_transient_image(desc));
            }
        }

        let history_idx = (self.current_frame + entry.len() - 1) % entry.len();
        entry[history_idx]
    }
}
