use super::*;

#[test]
fn test_image_handle_equality() {
    let h1 = ImageHandle(SymbolId(1));
    let h2 = ImageHandle(SymbolId(1));
    let h3 = ImageHandle(SymbolId(2));
    assert_eq!(h1, h2);
    assert_ne!(h1, h3);
}

#[test]
fn test_resource_usage_constants() {
    assert_ne!(ResourceUsage::READ, ResourceUsage::WRITE);
}
