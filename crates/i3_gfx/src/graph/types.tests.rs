use super::*;

#[test]
fn test_handle_equality() {
    let id1 = ImageHandle::new(1);
    let id2 = ImageHandle::new(1);
    let id3 = BufferHandle::new(1);
    assert_eq!(id1, id2);
    // Note: ImageHandle and BufferHandle are different types,
    // so id1 == id3 won't even compile (which is what we want)
}

#[test]
fn test_resource_usage_constants() {
    assert_ne!(ResourceUsage::READ, ResourceUsage::WRITE);
}
