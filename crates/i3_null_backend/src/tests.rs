use super::*;
use i3_gfx::graph::backend::{BackendImage, ImageDesc, RenderBackend};

#[test]
fn test_null_backend_existence_validation() {
    let mut backend = NullBackend::new();

    let desc = ImageDesc::new(1024, 1024, i3_gfx::graph::types::Format::R8G8B8A8_UNORM);
    let _valid_handle = backend.create_image(&desc);
    let _invalid_handle = BackendImage(999);

    let buf_desc = i3_gfx::graph::backend::BufferDesc {
        size: 1024,
        usage: i3_gfx::graph::types::BufferUsageFlags::VERTEX_BUFFER,
        memory: i3_gfx::graph::types::MemoryType::GpuOnly,
    };
    let valid_buffer = backend.create_buffer(&buf_desc);

    let pipelines = HashSet::new();
    let mut ctx = NullPassContext::new(
        "TestPass",
        &backend.allocated_images,
        &backend.allocated_buffers,
        &pipelines,
        &backend.image_map,
        1,
    );

    ctx.bind_vertex_buffer(
        0,
        i3_gfx::graph::types::BufferHandle(i3_gfx::graph::types::SymbolId(valid_buffer.0)),
    );
    assert!(ctx.failures().is_empty());

    // Invalid bind
    ctx.bind_vertex_buffer(
        1,
        i3_gfx::graph::types::BufferHandle(i3_gfx::graph::types::SymbolId(999)),
    );
    assert_eq!(ctx.failures().len(), 1);
    match &ctx.failures()[0] {
        ValidationError::ResourceNotFound(h) => assert_eq!(*h, 999),
        _ => panic!("Expected ResourceNotFound"),
    }
}
