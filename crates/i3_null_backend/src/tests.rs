use super::*;
use i3_gfx::graph::backend::{BackendImage, ImageDesc, RenderBackend};

#[test]
fn test_null_backend_existence_validation() {
    let mut backend = NullBackend::new();

    let desc = ImageDesc {
        width: 1024,
        height: 1024,
        format: 0,
    };
    let valid_handle = backend.create_image(&desc);
    let invalid_handle = BackendImage(999);

    let pipelines = HashSet::new();
    let mut ctx = NullPassContext::new(
        "TestPass",
        &backend.allocated_images,
        &backend.allocated_buffers,
        &pipelines,
    );

    // Valid bind
    ctx.bind_image(
        0,
        i3_gfx::graph::types::ImageHandle(i3_gfx::graph::types::SymbolId(valid_handle.0)),
    );
    assert!(ctx.failures().is_empty());

    // Invalid bind
    ctx.bind_image(
        1,
        i3_gfx::graph::types::ImageHandle(i3_gfx::graph::types::SymbolId(invalid_handle.0)),
    );
    assert_eq!(ctx.failures().len(), 1);
    match &ctx.failures()[0] {
        ValidationError::ResourceNotFound(h) => assert_eq!(*h, 999),
        _ => panic!("Expected ResourceNotFound"),
    }
}
