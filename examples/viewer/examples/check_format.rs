use i3_gfx::graph::types::Format;

fn main() {
    println!("Undefined: {}", Format::Undefined as u32);
    println!("R8G8B8A8_UNORM: {}", Format::R8G8B8A8_UNORM as u32);
    println!("R8G8B8A8_SRGB: {}", Format::R8G8B8A8_SRGB as u32);
    println!("B8G8R8A8_UNORM: {}", Format::B8G8R8A8_UNORM as u32);
    println!("B8G8R8A8_SRGB: {}", Format::B8G8R8A8_SRGB as u32);
    println!("R8G8_UNORM: {}", Format::R8G8_UNORM as u32);
    println!("R16G16_SFLOAT: {}", Format::R16G16_SFLOAT as u32);
    println!("R16G16B16A16_SFLOAT: {}", Format::R16G16B16A16_SFLOAT as u32);
    println!("R11G11B10_UFLOAT: {}", Format::R11G11B10_UFLOAT as u32);
    println!("R32_FLOAT: {}", Format::R32_FLOAT as u32);
    println!("R32G32B32A32_FLOAT: {}", Format::R32G32B32A32_FLOAT as u32);
    println!("D32_FLOAT: {}", Format::D32_FLOAT as u32);
    println!("R32_SFLOAT: {}", Format::R32_SFLOAT as u32);
}
