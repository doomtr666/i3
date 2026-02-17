use super::*;
use i3_gfx::graph::backend::{BackendImage, ImageDesc, RenderBackend};

#[test]
fn test_null_backend_existence_validation() {
    let mut backend = NullBackend::new();

    let desc = ImageDesc::new(1024, 1024, i3_gfx::graph::types::Format::R8G8B8A8_UNORM);
    let valid_handle = backend.create_image(&desc);
    let invalid_handle = BackendImage(999);

    let pipelines = HashSet::new();
    let mut ctx = NullPassContext::new(
        "TestPass",
        &backend.allocated_images,
        &backend.allocated_buffers,
        &pipelines,
        &backend.image_map,
    );

    ctx.bind_image(
        0, // Assuming a binding point
        i3_gfx::graph::types::ImageHandle(i3_gfx::graph::types::SymbolId(valid_handle.0)),
    );
    assert!(ctx.failures().is_empty());

    // Invalid bind
        1,
        i3_gfx::graph::types::ImageHandle(i3_gfx::graph::types::SymbolId(invalid_handle.0)),
    );
    assert_eq!(ctx.failures().len(), 1);
    match &ctx.failures()[0] {
        ValidationError::ResourceNotFound(h) => assert_eq!(*h, 999),
        _ => panic!("Expected ResourceNotFound"),
    }
}
