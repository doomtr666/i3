use std::any::{Any, TypeId};
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};

use crate::graph::backend::{BackendAccelerationStructure, BackendBuffer, BackendImage, DescriptorWrite};
use crate::graph::pass::{InternalPassBuilder, PassBuilder, RenderPass};
use crate::graph::symbol_table::{Symbol, SymbolTable};
use crate::graph::types::*;

pub(crate) static NEXT_NODE_ID: AtomicU64 = AtomicU64::new(1);

// ─────────────────────────────────────────────────────────────────────────────
// NodeStorage
// ─────────────────────────────────────────────────────────────────────────────

/// Storage for a specific node and its children.
pub struct NodeStorage {
    pub node_id: u64,
    pub name: String,
    pub symbols: SymbolTable,
    pub children: Vec<NodeStorage>,
    pub pass: Option<Box<dyn RenderPass>>,

    pub pipeline: Option<PipelineHandle>,

    // Captured resource intents (for leaf nodes)
    pub image_reads: Vec<(ImageHandle, ResourceUsage)>,
    pub image_writes: Vec<(ImageHandle, ResourceUsage)>,
    pub buffer_reads: Vec<(BufferHandle, ResourceUsage)>,
    pub buffer_writes: Vec<(BufferHandle, ResourceUsage)>,

    /// CPU data symbols published by this node (write dependency).
    pub data_writes: Vec<String>,
    /// CPU data symbols consumed by this node (read dependency).
    pub data_reads: Vec<String>,

    pub external_images: Vec<(ImageHandle, BackendImage)>,
    pub external_buffers: Vec<(BufferHandle, BackendBuffer)>,
    pub external_accel_structs: Vec<(AccelerationStructureHandle, BackendAccelerationStructure)>,
    pub swapchain_requests: Vec<(ImageHandle, WindowHandle)>,
    pub descriptor_sets: Vec<(u32, Vec<DescriptorWrite>)>,
    pub prefer_async: bool,
    /// Images to transition to PresentSrc AFTER this pass executes (post-transition).
    pub present_images: Vec<ImageHandle>,
}

impl std::fmt::Debug for NodeStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeStorage")
            .field("name", &self.name)
            .field("symbols", &self.symbols)
            .field("children", &self.children)
            .field("image_reads", &self.image_reads)
            .field("image_writes", &self.image_writes)
            .field("external_images", &self.external_images)
            .field("external_buffers", &self.external_buffers)
            .field("swapchain_requests", &self.swapchain_requests)
            .finish()
    }
}

impl NodeStorage {
    pub fn new(node_id: u64, name: impl Into<String>, prefer_async: bool) -> Self {
        Self {
            node_id,
            name: name.into(),
            symbols: SymbolTable::new(),
            children: Vec::new(),
            pass: None,
            pipeline: None,
            image_reads: Vec::new(),
            image_writes: Vec::new(),
            buffer_reads: Vec::new(),
            buffer_writes: Vec::new(),
            data_writes: Vec::new(),
            data_reads: Vec::new(),
            external_images: Vec::new(),
            external_buffers: Vec::new(),
            external_accel_structs: Vec::new(),
            swapchain_requests: Vec::new(),
            descriptor_sets: Vec::new(),
            prefer_async,
            present_images: Vec::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PassRecorder — InternalPassBuilder impl
// ─────────────────────────────────────────────────────────────────────────────

pub struct PassRecorder<'a> {
    pub storage: &'a mut NodeStorage,
    pub ancestor_symbols: Vec<&'a SymbolTable>,
}

impl<'a> PassRecorder<'a> {
    fn declare_image_impl(
        &mut self,
        name: &str,
        desc: ImageDesc,
        lifetime: SymbolLifetime,
        is_output: bool,
    ) -> ImageHandle {
        let index = self.storage.symbols.symbols.len() as u64;
        let id = SymbolId((self.storage.node_id << 32) | index);
        let actual_handle = ImageHandle(id);
        self.storage.symbols.publish(
            name,
            Symbol {
                name: name.to_string(),
                symbol_type: SymbolType::Image(desc),
                lifetime,
                data: Some(Arc::new(actual_handle)),
                is_output,
            },
            id,
        );
        actual_handle
    }

    fn declare_buffer_impl(
        &mut self,
        name: &str,
        desc: BufferDesc,
        lifetime: SymbolLifetime,
        is_output: bool,
    ) -> BufferHandle {
        let index = self.storage.symbols.symbols.len() as u64;
        let id = SymbolId((self.storage.node_id << 32) | index);
        let actual_handle = BufferHandle(id);
        self.storage.symbols.publish(
            name,
            Symbol {
                name: name.to_string(),
                symbol_type: SymbolType::Buffer(desc),
                lifetime,
                data: Some(Arc::new(actual_handle)),
                is_output,
            },
            id,
        );
        actual_handle
    }
}

impl<'a> InternalPassBuilder for PassRecorder<'a> {
    fn publish_erased(&mut self, _type_id: TypeId, name: &str, data: Box<dyn Any + Send + Sync>) {
        tracing::trace!(name, "Publishing CPU data");
        let index = self.storage.symbols.symbols.len() as u64;
        let id = SymbolId((self.storage.node_id << 32) | index);
        self.storage.symbols.publish(
            name,
            Symbol {
                name: name.to_string(),
                symbol_type: SymbolType::CpuData(_type_id),
                lifetime: SymbolLifetime::Transient,
                data: Some(Arc::from(data)),
                is_output: false,
            },
            id,
        );
        self.storage.data_writes.push(name.to_string());
    }

    fn consume_erased(&mut self, type_id: TypeId, name: &str) -> &dyn Any {
        self.try_consume_erased(type_id, name)
            .unwrap_or_else(|| panic!("Symbol '{}' not found in current or parent scope", name))
    }

    fn try_consume_erased(&mut self, _type_id: TypeId, name: &str) -> Option<&dyn Any> {
        self.storage.data_reads.push(name.to_string());

        if let Some(id) = self.storage.symbols.resolve(name) {
            tracing::trace!(name, "Consuming CPU data (local)");
            return self.storage.symbols.get_data(id);
        }

        for parent in self.ancestor_symbols.iter().rev() {
            if let Some(id) = parent.resolve(name) {
                tracing::trace!(name, "Consuming CPU data (inherited)");
                return parent.get_data(id);
            }
        }

        None
    }

    fn get_image_desc(&self, handle: ImageHandle) -> ImageDesc {
        self.storage.symbols.symbols.iter()
            .find(|s| matches!(s.symbol_type, SymbolType::Image(_)) && s.data.as_ref().and_then(|d| d.downcast_ref::<ImageHandle>()) == Some(&handle))
            .and_then(|s| if let SymbolType::Image(desc) = s.symbol_type { Some(desc) } else { None })
            .unwrap_or_else(|| {
                for parent in self.ancestor_symbols.iter().rev() {
                    if let Some(s) = parent.symbols.iter().find(|s| matches!(s.symbol_type, SymbolType::Image(_)) && s.data.as_ref().and_then(|d| d.downcast_ref::<ImageHandle>()) == Some(&handle)) {
                        if let SymbolType::Image(desc) = s.symbol_type { return desc; }
                    }
                }
                ImageDesc::default()
            })
    }

    fn get_buffer_desc(&self, handle: BufferHandle) -> crate::graph::types::BufferDesc {
        self.storage.symbols.symbols.iter()
            .find(|s| matches!(s.symbol_type, SymbolType::Buffer(_)) && s.data.as_ref().and_then(|d| d.downcast_ref::<BufferHandle>()) == Some(&handle))
            .and_then(|s| if let SymbolType::Buffer(desc) = s.symbol_type { Some(desc) } else { None })
            .unwrap_or_else(|| {
                for parent in self.ancestor_symbols.iter().rev() {
                    if let Some(s) = parent.symbols.iter().find(|s| matches!(s.symbol_type, SymbolType::Buffer(_)) && s.data.as_ref().and_then(|d| d.downcast_ref::<BufferHandle>()) == Some(&handle)) {
                        if let SymbolType::Buffer(desc) = s.symbol_type { return desc; }
                    }
                }
                BufferDesc::default()
            })
    }

    fn read_image(&mut self, handle: ImageHandle, usage: ResourceUsage) {
        self.storage.image_reads.push((handle, usage));
    }

    fn write_image(&mut self, handle: ImageHandle, usage: ResourceUsage) {
        self.storage.image_writes.push((handle, usage));
    }

    fn declare_present_image(&mut self, handle: ImageHandle) {
        self.storage.present_images.push(handle);
        self.storage
            .image_reads
            .push((handle, ResourceUsage::PRESENT));
    }

    fn bind_pipeline(&mut self, handle: PipelineHandle) {
        self.storage.pipeline = Some(handle);
    }

    fn bind_descriptor_set(&mut self, set_index: u32, writes: Vec<DescriptorWrite>) {
        self.storage.descriptor_sets.push((set_index, writes));
    }

    fn register_external_image(&mut self, handle: ImageHandle, physical: BackendImage) {
        self.storage.external_images.push((handle, physical));
    }

    fn register_external_buffer(&mut self, handle: BufferHandle, physical: BackendBuffer) {
        self.storage.external_buffers.push((handle, physical));
    }

    fn read_buffer(&mut self, handle: BufferHandle, usage: ResourceUsage) {
        self.storage.buffer_reads.push((handle, usage));
    }

    fn write_buffer(&mut self, handle: BufferHandle, usage: ResourceUsage) {
        self.storage.buffer_writes.push((handle, usage));
    }

    fn declare_image(&mut self, name: &str, desc: ImageDesc) -> ImageHandle {
        self.declare_image_impl(name, desc, SymbolLifetime::Transient, false)
    }

    fn declare_image_output(&mut self, name: &str, desc: ImageDesc) -> ImageHandle {
        self.declare_image_impl(name, desc, SymbolLifetime::Transient, true)
    }

    fn declare_image_history(&mut self, name: &str, desc: ImageDesc) -> ImageHandle {
        self.declare_image_impl(name, desc, SymbolLifetime::TemporalHistory, false)
    }

    fn declare_image_history_output(&mut self, name: &str, desc: ImageDesc) -> ImageHandle {
        self.declare_image_impl(name, desc, SymbolLifetime::TemporalHistory, true)
    }

    fn read_image_history(&mut self, name: &str) -> ImageHandle {
        let history_name = format!("{}_History", name);
        self.declare_image_impl(
            &history_name,
            ImageDesc::default(),
            SymbolLifetime::TemporalHistory,
            false,
        )
    }

    fn declare_buffer(&mut self, name: &str, desc: BufferDesc) -> BufferHandle {
        self.declare_buffer_impl(name, desc, SymbolLifetime::Transient, false)
    }

    fn declare_buffer_output(&mut self, name: &str, desc: BufferDesc) -> BufferHandle {
        self.declare_buffer_impl(name, desc, SymbolLifetime::Transient, true)
    }

    fn declare_buffer_history(&mut self, name: &str, desc: BufferDesc) -> BufferHandle {
        self.declare_buffer_impl(name, desc, SymbolLifetime::TemporalHistory, false)
    }

    fn declare_buffer_history_output(&mut self, name: &str, desc: BufferDesc) -> BufferHandle {
        self.declare_buffer_impl(name, desc, SymbolLifetime::TemporalHistory, true)
    }

    fn read_buffer_history(&mut self, name: &str) -> BufferHandle {
        let history_name = format!("{}_History", name);
        self.declare_buffer_impl(
            &history_name,
            BufferDesc {
                size: 0,
                usage: BufferUsageFlags::empty(),
                memory: MemoryType::GpuOnly,
            },
            SymbolLifetime::TemporalHistory,
            false,
        )
    }

    fn import_buffer(&mut self, name: &str, physical: BackendBuffer) -> BufferHandle {
        let index = self.storage.symbols.symbols.len() as u64;
        let id = SymbolId((self.storage.node_id << 32) | index);
        let actual_handle = BufferHandle(id);
        self.storage.symbols.publish(
            name,
            Symbol {
                name: name.to_string(),
                symbol_type: SymbolType::Buffer(BufferDesc {
                    size: 0,
                    usage: BufferUsageFlags::empty(),
                    memory: MemoryType::GpuOnly,
                }),
                lifetime: SymbolLifetime::External,
                data: Some(Arc::new(actual_handle)),
                is_output: true,
            },
            id,
        );
        self.register_external_buffer(actual_handle, physical);
        actual_handle
    }

    fn import_acceleration_structure(
        &mut self,
        name: &str,
        physical: BackendAccelerationStructure,
    ) -> AccelerationStructureHandle {
        let index = self.storage.symbols.symbols.len() as u64;
        let id = SymbolId((self.storage.node_id << 32) | index);
        let handle = AccelerationStructureHandle(id);
        self.storage.symbols.publish(
            name,
            Symbol {
                name: name.to_string(),
                symbol_type: SymbolType::AccelStruct(AccelerationStructureDesc { size: 0 }),
                lifetime: SymbolLifetime::External,
                data: Some(Arc::new(handle)),
                is_output: true,
            },
            id,
        );
        self.storage.external_accel_structs.push((handle, physical));
        handle
    }

    fn try_resolve_acceleration_structure(
        &mut self,
        name: &str,
    ) -> Option<AccelerationStructureHandle> {
        // Check local scope first, then ancestor scopes.
        if let Some(id) = self.storage.symbols.resolve(name) {
            if let Some(data) = self.storage.symbols.get_data(id) {
                return data.downcast_ref::<AccelerationStructureHandle>().copied();
            }
        }
        for parent in self.ancestor_symbols.iter().rev() {
            if let Some(id) = parent.resolve(name) {
                if let Some(data) = parent.get_data(id) {
                    return data.downcast_ref::<AccelerationStructureHandle>().copied();
                }
            }
        }
        None
    }

    fn acquire_backbuffer(&mut self, window: WindowHandle) -> ImageHandle {
        let name = format!("Window_{}", window.0);
        let index = self.storage.symbols.symbols.len() as u64;
        let id = SymbolId((self.storage.node_id << 32) | index);
        self.storage.symbols.publish(
            &name,
            Symbol {
                name: name.clone(),
                symbol_type: SymbolType::Image(ImageDesc {
                    width: 1280,
                    height: 720,
                    depth: 1,
                    format: Format::B8G8R8A8_SRGB,
                    mip_levels: 1,
                    array_layers: 1,
                    usage: ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::TRANSFER_DST,
                    view_type: ImageViewType::Type2D,
                    swizzle: ComponentMapping::default(),
                    clear_value: None,
                }),
                lifetime: SymbolLifetime::External,
                data: None,
                is_output: false,
            },
            id,
        );
        let actual_handle = ImageHandle(id);
        self.storage.symbols.symbols[index as usize].data = Some(Arc::new(actual_handle));
        self.storage.swapchain_requests.push((actual_handle, window));
        actual_handle
    }

    fn add_node_erased(&mut self, node: Box<dyn RenderPass>) {
        tracing::trace!(name = node.name(), "Adding sub-node");

        let prefer_async = node.prefer_async();
        let mut child_storage = NodeStorage::new(
            NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed),
            node.name(),
            prefer_async,
        );
        child_storage.pass = Some(node);

        let mut pass = child_storage.pass.take().unwrap();

        {
            let mut ancestors = self.ancestor_symbols.clone();
            ancestors.push(&self.storage.symbols);

            let mut sub_recorder = PassRecorder {
                storage: &mut child_storage,
                ancestor_symbols: ancestors,
            };
            let mut builder = PassBuilder { inner: &mut sub_recorder };
            pass.declare(&mut builder);
        }

        // Propagate output symbols into the parent scope.
        for symbol in &child_storage.symbols.symbols {
            if symbol.is_output {
                let index = self.storage.symbols.symbols.len() as u64;
                let id = SymbolId((self.storage.node_id << 32) | index);
                self.storage.symbols.publish(
                    &symbol.name,
                    Symbol {
                        name: symbol.name.clone(),
                        symbol_type: symbol.symbol_type.clone(),
                        lifetime: symbol.lifetime,
                        data: symbol.data.clone(),
                        is_output: symbol.is_output,
                    },
                    id,
                );
            }
        }

        child_storage.pass = Some(pass);
        self.storage.children.push(child_storage);
    }
}
