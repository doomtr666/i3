pub const SOA_WIDTH: usize = 8;

#[repr(C, align(32))]
#[derive(Clone, Copy, Default)]
pub struct NoisePoint {
    pub x: [f32; SOA_WIDTH],
    pub y: [f32; SOA_WIDTH],
    pub z: [f32; SOA_WIDTH],
    pub dx: [f32; SOA_WIDTH],
    pub dy: [f32; SOA_WIDTH],
    pub dz: [f32; SOA_WIDTH],
}

#[repr(C, align(32))]
#[derive(Clone, Copy, Default)]
pub struct NoiseSample {
    pub value: [f32; SOA_WIDTH],
    pub dx: [f32; SOA_WIDTH],
    pub dy: [f32; SOA_WIDTH],
    pub dz: [f32; SOA_WIDTH],
}

pub trait Generator {
    fn sample(&self, point: &NoisePoint) -> NoiseSample;
}

impl<T: Generator + ?Sized> Generator for std::boxed::Box<T> {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        (**self).sample(point)
    }
}

impl<T: Generator + ?Sized> Generator for std::sync::Arc<T> {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        (**self).sample(point)
    }
}
