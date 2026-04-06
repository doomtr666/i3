use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;
use crate::graph::types::{SymbolId, SymbolLifetime, SymbolType};

// ─────────────────────────────────────────────────────────────────────────────
// FrameBlackboard
// ─────────────────────────────────────────────────────────────────────────────

/// Type-keyed map for per-frame CPU data.
/// Published by the caller of `CompiledGraph::execute()` and consumed by
/// `PassContext::consume_frame()` inside `RenderPass::execute()`.
pub struct FrameBlackboard {
    data: HashMap<(&'static str, TypeId), Arc<dyn Any + Send + Sync>>,
}

impl FrameBlackboard {
    pub fn new() -> Self {
        Self { data: HashMap::new() }
    }

    /// Publish a typed value under `name` for this frame.
    pub fn publish<T: 'static + Send + Sync>(&mut self, name: &'static str, data: T) {
        self.data.insert((name, TypeId::of::<T>()), Arc::new(data));
    }

    /// Retrieve a reference to the value. Panics if not found.
    pub fn consume<T: 'static + Send + Sync>(&self, name: &'static str) -> &T {
        self.try_consume::<T>(name)
            .unwrap_or_else(|| panic!("FrameBlackboard: '{}' not found", name))
    }

    /// Retrieve a reference to the value. Returns None if not found.
    pub fn try_consume<T: 'static + Send + Sync>(&self, name: &'static str) -> Option<&T> {
        self.data
            .get(&(name, TypeId::of::<T>()))
            .and_then(|arc| arc.downcast_ref::<T>())
    }
}

impl Default for FrameBlackboard {
    fn default() -> Self { Self::new() }
}

// ─────────────────────────────────────────────────────────────────────────────
// Symbol + SymbolTable
// ─────────────────────────────────────────────────────────────────────────────

/// Metadata and data for an entry in the symbol table.
pub struct Symbol {
    pub name: String,
    pub symbol_type: SymbolType,
    pub lifetime: SymbolLifetime,
    /// Arc so that output symbols can be cloned into the parent scope cheaply.
    pub data: Option<Arc<dyn Any + Send + Sync>>,
    /// If true, this symbol is promoted to the parent scope after declare() completes.
    /// Used by passes acting as groups to export transient resources to siblings.
    pub is_output: bool,
}

impl std::fmt::Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Symbol")
            .field("name", &self.name)
            .field("symbol_type", &self.symbol_type)
            .field("lifetime", &self.lifetime)
            .field("has_data", &self.data.is_some())
            .field("is_output", &self.is_output)
            .finish()
    }
}

/// A scope-local symbol table.
#[derive(Debug)]
pub struct SymbolTable {
    pub(crate) symbols: Vec<Symbol>,
    name_to_id: HashMap<String, SymbolId>,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            symbols: Vec::new(),
            name_to_id: HashMap::new(),
        }
    }

    pub fn publish(&mut self, name: &str, symbol: Symbol, id: SymbolId) -> SymbolId {
        self.name_to_id.insert(name.to_string(), id);
        self.symbols.push(symbol);
        id
    }

    pub fn resolve(&self, name: &str) -> Option<SymbolId> {
        self.name_to_id.get(name).copied()
    }

    pub fn get_data(&self, id: SymbolId) -> Option<&dyn Any> {
        let index = (id.0 & 0xFFFFFFFF) as usize;
        self.symbols
            .get(index)
            .and_then(|s| s.data.as_ref().map(|d| d.as_ref() as &dyn Any))
    }
}
