use i3_gfx::graph::types::Format;
use i3_vulkan_backend::convert::{convert_format, convert_u32_format};

fn main() {
    for i in 0..24 {
        let u32_fmt = convert_u32_format(i as u32);
        // We can't easily get the Format variant from u32 here without a long match
        // but we can check a few key ones.
        println!("Index {}: {:?}", i, u32_fmt);
    }
}
