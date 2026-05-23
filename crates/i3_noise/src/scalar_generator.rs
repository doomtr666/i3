#[repr(C)]
#[derive(Default, Copy, Clone)]
pub struct ScalarPosition {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C)]
#[derive(Default, Copy, Clone)]
pub struct ScalarSample {
    pub value: f32,
    pub dx: f32,
    pub dy: f32,
    pub dz: f32,
}

pub trait ScalarGenerator {
    fn eval_scalar(&self, pos: &ScalarPosition, dd: f32) -> ScalarSample;
}
