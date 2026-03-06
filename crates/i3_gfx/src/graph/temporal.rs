use crate::graph::backend::{BackendBuffer, BackendImage, RenderBackendInternal};
use crate::graph::types::{BufferDesc, ImageDesc};
use std::collections::HashMap;

/// Registry that persists across frames to manage temporal history resources.
/// It holds double-buffered physical Vulkan resources based on their graph name.
pub struct TemporalRegistry {
    // Current logical frame flip-flop (0 or 1). Determines which is "current" and which is "history".
    pub current_frame: usize,

    // We store arrays of 2 physical resources.
    // Index `current_frame` is the target for this frame.
    // Index `1 - current_frame` is the history from the previous frame.
    pub(crate) buffers: HashMap<String, [BackendBuffer; 2]>,
    pub(crate) images: HashMap<String, [BackendImage; 2]>,
}

impl Default for TemporalRegistry {
    fn default() -> Self {
        Self {
            current_frame: 0,
            buffers: HashMap::new(),
            images: HashMap::new(),
        }
    }
}

impl TemporalRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Called at the beginning of a frame or end of a frame to flip history buffers.
    pub fn advance_frame(&mut self) {
        self.current_frame = 1 - self.current_frame;
    }

    /// Retrieve or create a persistent double-buffered physical buffer.
    pub fn get_or_create_buffer<B: RenderBackendInternal>(
        &mut self,
        name: &str,
        desc: &BufferDesc,
        backend: &mut B,
    ) -> BackendBuffer {
        let entry = self.buffers.entry(name.to_string()).or_insert_with(|| {
            let buf0 = backend.create_transient_buffer(desc);
            let buf1 = backend.create_transient_buffer(desc);
            [buf0, buf1]
        });
        entry[self.current_frame].clone()
    }

    /// Retrieve the history buffer (N-1) for a persistent double-buffered physical buffer.
    /// If it hasn't been created yet, this will also create it.
    pub fn get_or_create_history_buffer<B: RenderBackendInternal>(
        &mut self,
        name: &str,
        desc: &BufferDesc,
        backend: &mut B,
    ) -> BackendBuffer {
        let entry = self.buffers.entry(name.to_string()).or_insert_with(|| {
            let buf0 = backend.create_transient_buffer(desc);
            let buf1 = backend.create_transient_buffer(desc);
            [buf0, buf1]
        });
        entry[1 - self.current_frame].clone()
    }

    /// Retrieve or create a persistent double-buffered physical image.
    pub fn get_or_create_image<B: RenderBackendInternal>(
        &mut self,
        name: &str,
        desc: &ImageDesc,
        backend: &mut B,
    ) -> BackendImage {
        let entry = self.images.entry(name.to_string()).or_insert_with(|| {
            let img0 = backend.create_transient_image(desc);
            let img1 = backend.create_transient_image(desc);
            [img0, img1]
        });
        entry[self.current_frame]
    }

    /// Retrieve the history image (N-1).
    pub fn get_or_create_history_image<B: RenderBackendInternal>(
        &mut self,
        name: &str,
        desc: &ImageDesc,
        backend: &mut B,
    ) -> BackendImage {
        let entry = self.images.entry(name.to_string()).or_insert_with(|| {
            let img0 = backend.create_transient_image(desc);
            let img1 = backend.create_transient_image(desc);
            [img0, img1]
        });
        entry[1 - self.current_frame]
    }
}
