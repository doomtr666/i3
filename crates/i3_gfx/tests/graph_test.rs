use i3_gfx::graph::pass::{PassBuilder, RenderPass};
use i3_gfx::graph::types::{BufferHandle, ImageHandle, PassDomain, ResourceUsage};

struct TestPass;

impl RenderPass for TestPass {
    fn name(&self) -> &str {
        "TestPass"
    }
    fn domain(&self) -> PassDomain {
        PassDomain::Graphics
    }
    fn declare(&self, builder: &mut dyn PassBuilder) {
        let input = ImageHandle::new(1);
        let output = ImageHandle::new(2);
        builder.read_image(input, ResourceUsage::READ);
        builder.write_image(
            output,
            ResourceUsage::WRITE | ResourceUsage::COLOR_ATTACHMENT,
        );
    }
}

struct MockBuilder;
impl PassBuilder for MockBuilder {
    fn read_image(&mut self, _handle: ImageHandle, _usage: ResourceUsage) {}
    fn write_image(&mut self, _handle: ImageHandle, _usage: ResourceUsage) {}
    fn read_buffer(&mut self, _handle: BufferHandle, _usage: ResourceUsage) {}
    fn write_buffer(&mut self, _handle: BufferHandle, _usage: ResourceUsage) {}
}

#[test]
fn test_pass_declaration_syntax() {
    let pass = TestPass;
    let mut builder = MockBuilder;
    pass.declare(&mut builder);
    assert_eq!(pass.name(), "TestPass");
}
