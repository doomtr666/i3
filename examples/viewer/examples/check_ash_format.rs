use ash::vk;

fn main() {
    println!("B10G11R11_UFLOAT_PACK32: {:?}", vk::Format::B10G11R11_UFLOAT_PACK32);
    // Let's see if R11G11B10 exists
    // println!("R11G11B10_UFLOAT_PACK32: {:?}", vk::Format::R11G11B10_UFLOAT_PACK32);
}
