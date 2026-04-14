use crate::graph::backend::{BackendBuffer, BackendImage, RenderBackendInternal};
use crate::graph::types::{BufferDesc, ImageDesc};
use std::collections::HashMap;

struct TemporalImageEntry {
    images: Vec<BackendImage>,
    /// The ImageDesc used when these images were allocated.
    desc: ImageDesc,
    /// Images queued for deferred destruction after a resize.
    /// Each element: (monotonic frame index at which it is safe to destroy, image).
    /// Images are held alive for `capacity` frames to avoid destroying resources
    /// that are still referenced by in-flight GPU work.
    pending_delete: Vec<(usize, BackendImage)>,
}

/// Registry that persists across frames to manage temporal history resources.
/// It holds double-buffered physical Vulkan resources based on their graph name.
pub struct TemporalRegistry {
    // Current logical frame index (0..capacity-1), wraps around.
    pub current_frame: usize,
    // Maximum number of frames in flight (usually matches swapchain image count).
    pub capacity: usize,
    // Monotonically increasing frame counter, used for deferred image deletion timing.
    frame_count: usize,

    pub(crate) buffers: HashMap<String, Vec<BackendBuffer>>,
    images: HashMap<String, TemporalImageEntry>,
}

impl Default for TemporalRegistry {
    fn default() -> Self {
        Self {
            current_frame: 0,
            capacity: 2,
            frame_count: 0,
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
        self.frame_count += 1;
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

    /// Retrieve or create a persistent N-buffered physical image (producer side).
    ///
    /// This is the authoritative call that owns the image pool and its `desc`.
    /// When `desc` dimensions/format/mips differ from the stored images (e.g. after a
    /// window resize), old images are queued for deferred destruction: they are actually
    /// destroyed only after `capacity` more frames, ensuring no in-flight GPU access.
    pub fn get_or_create_image<B: RenderBackendInternal>(
        &mut self,
        name: &str,
        desc: &ImageDesc,
        backend: &mut B,
    ) -> BackendImage {
        let cap = self.capacity;
        let frame_count = self.frame_count;

        let entry = self.images.entry(name.to_string()).or_insert_with(|| {
            TemporalImageEntry {
                images: (0..cap)
                    .map(|_| backend.create_transient_image(desc))
                    .collect(),
                desc: *desc,
                pending_delete: Vec::new(),
            }
        });

        // Flush images whose deferred-delete deadline has passed.
        Self::flush_pending(entry, frame_count, backend);

        // Detect dimension/format/mip changes (e.g. window resize).
        // Old images are queued for deferred destruction so they stay alive until the
        // GPU frames that reference them have completed.
        if entry.desc.width != desc.width
            || entry.desc.height != desc.height
            || entry.desc.format != desc.format
            || entry.desc.mip_levels != desc.mip_levels
        {
            let safe_after = frame_count + cap;
            for img in entry.images.drain(..) {
                entry.pending_delete.push((safe_after, img));
            }
            entry.desc = *desc;
        }

        // Grow pool if capacity increased or images were cleared by resize.
        while entry.images.len() < cap {
            entry.images.push(backend.create_transient_image(desc));
        }

        entry.images[self.current_frame % entry.images.len()]
    }

    /// Retrieve the history image (N-1) (consumer side).
    ///
    /// The image pool and its desc are owned by `get_or_create_image`. This function
    /// only reads from the existing pool. The `desc` parameter may be `ImageDesc::default()`
    /// (the consumer doesn't know the real size), so it is intentionally ignored for
    /// sizing decisions — the stored `entry.desc` is always used instead.
    pub fn get_or_create_history_image<B: RenderBackendInternal>(
        &mut self,
        name: &str,
        _desc: &ImageDesc,
        backend: &mut B,
    ) -> BackendImage {
        let cap = self.capacity;
        let frame_count = self.frame_count;

        // The entry must already exist, created by the matching get_or_create_image call.
        // If it doesn't (edge case: consumer declared before producer), we have no valid
        // desc to create images from — return INVALID and log a warning.
        let Some(entry) = self.images.get_mut(name) else {
            tracing::warn!(
                "get_or_create_history_image('{}') called before get_or_create_image — no pool exists yet",
                name
            );
            return BackendImage(u64::MAX);
        };

        Self::flush_pending(entry, frame_count, backend);

        // Grow pool using the stored desc (always valid), not the caller-supplied one.
        while entry.images.len() < cap {
            let desc = entry.desc;
            entry.images.push(backend.create_transient_image(&desc));
        }

        let history_idx = (self.current_frame + entry.images.len() - 1) % entry.images.len();
        entry.images[history_idx]
    }

    /// Destroy images whose deferred-delete deadline has passed.
    fn flush_pending<B: RenderBackendInternal>(
        entry: &mut TemporalImageEntry,
        frame_count: usize,
        backend: &mut B,
    ) {
        entry.pending_delete.retain(|&(safe_after, img)| {
            if frame_count >= safe_after {
                backend.destroy_image(img);
                false
            } else {
                true
            }
        });
    }
}
