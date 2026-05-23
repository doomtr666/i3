use crate::scalar_generator::{ScalarGenerator, ScalarPosition, ScalarSample};

#[repr(C, align(32))]
#[derive(Default, Copy, Clone)]
pub struct VecPosition {
    pub x: [f32; 8],
    pub y: [f32; 8],
    pub z: [f32; 8],
}

#[repr(C, align(32))]
#[derive(Default, Copy, Clone)]
pub struct VecSample {
    pub value: [f32; 8],
    pub dx: [f32; 8],
    pub dy: [f32; 8],
    pub dz: [f32; 8],
}

pub trait VecGenerator {
    fn eval_vec(&self, pos: &VecPosition, dd: f32) -> VecSample;
}

impl<T> VecGenerator for T
where
    T: ScalarGenerator,
{
    #[inline(always)]
    fn eval_vec(&self, pos: &VecPosition, dd: f32) -> VecSample {
        let mut res = VecSample::default();

        for i in 0..8 {
            let scalar_pos = ScalarPosition {
                x: pos.x[i],
                y: pos.y[i],
                z: pos.z[i],
            };
            let scalar_res = self.eval_scalar(&scalar_pos, dd);

            res.value[i] = scalar_res.value;
            res.dx[i] = scalar_res.dx;
            res.dy[i] = scalar_res.dy;
            res.dz[i] = scalar_res.dz;
        }

        res
    }
}
