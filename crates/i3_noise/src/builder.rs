use crate::{Simplex, VecGenerator, VecPosition, VecSample};

#[cfg(target_arch = "x86_64")]
use crate::SimplexAvx;

#[derive(Clone)]
pub enum SimplexGenerator {
    Scalar(Simplex),
    #[cfg(target_arch = "x86_64")]
    Avx(SimplexAvx),
}

impl VecGenerator for SimplexGenerator {
    #[inline(always)]
    fn eval_vec(&self, pos: &VecPosition, dd: f32) -> VecSample {
        match self {
            SimplexGenerator::Scalar(s) => s.eval_vec(pos, dd),
            #[cfg(target_arch = "x86_64")]
            SimplexGenerator::Avx(a) => a.eval_vec(pos, dd),
        }
    }
}

pub struct SimplexBuilder {
    seed: u32,
    force_scalar: bool,
}

impl SimplexBuilder {
    pub fn new() -> Self {
        SimplexBuilder {
            seed: 0,
            force_scalar: false,
        }
    }

    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    pub fn force_scalar(mut self, force: bool) -> Self {
        self.force_scalar = force;
        self
    }

    pub fn build(self) -> SimplexGenerator {
        #[cfg(target_arch = "x86_64")]
        {
            if !self.force_scalar && std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return SimplexGenerator::Avx(SimplexAvx::new(self.seed));
            }
        }
        
        SimplexGenerator::Scalar(Simplex::new(self.seed))
    }
}

use crate::FbmVec;
#[cfg(target_arch = "x86_64")]
use crate::FbmAvx;

#[derive(Clone)]
pub enum FbmGenerator {
    Scalar(FbmVec<SimplexGenerator>),
    #[cfg(target_arch = "x86_64")]
    Avx(FbmAvx),
}

impl VecGenerator for FbmGenerator {
    #[inline(always)]
    fn eval_vec(&self, pos: &VecPosition, dd: f32) -> VecSample {
        match self {
            FbmGenerator::Scalar(s) => s.eval_vec(pos, dd),
            #[cfg(target_arch = "x86_64")]
            FbmGenerator::Avx(a) => a.eval_vec(pos, dd),
        }
    }
}

pub struct FbmBuilder {
    seed: u32,
    octaves: usize,
    lacunarity: f32,
    gain: f32,
    force_scalar: bool,
}

impl FbmBuilder {
    pub fn new() -> Self {
        FbmBuilder {
            seed: 0,
            octaves: 1,
            lacunarity: 2.0,
            gain: 0.5,
            force_scalar: false,
        }
    }

    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    pub fn octaves(mut self, octaves: usize) -> Self {
        self.octaves = octaves;
        self
    }

    pub fn lacunarity(mut self, lacunarity: f32) -> Self {
        self.lacunarity = lacunarity;
        self
    }

    pub fn gain(mut self, gain: f32) -> Self {
        self.gain = gain;
        self
    }

    pub fn force_scalar(mut self, force: bool) -> Self {
        self.force_scalar = force;
        self
    }

    pub fn build(self) -> FbmGenerator {
        #[cfg(target_arch = "x86_64")]
        {
            if !self.force_scalar && std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                let source = SimplexAvx::new(self.seed);
                return FbmGenerator::Avx(FbmAvx::new(source, self.octaves, self.lacunarity, self.gain));
            }
        }
        
        let source = SimplexBuilder::new().seed(self.seed).force_scalar(self.force_scalar).build();
        FbmGenerator::Scalar(FbmVec::new(source, self.octaves, self.lacunarity, self.gain))
    }
}

use crate::ErosionFbmVec;
#[cfg(target_arch = "x86_64")]
use crate::ErosionFbmAvx;

#[derive(Clone)]
pub enum ErosionFbmGenerator {
    Scalar(ErosionFbmVec<SimplexGenerator>),
    #[cfg(target_arch = "x86_64")]
    Avx(ErosionFbmAvx),
}

impl VecGenerator for ErosionFbmGenerator {
    #[inline(always)]
    fn eval_vec(&self, pos: &VecPosition, dd: f32) -> VecSample {
        match self {
            ErosionFbmGenerator::Scalar(s) => s.eval_vec(pos, dd),
            #[cfg(target_arch = "x86_64")]
            ErosionFbmGenerator::Avx(a) => a.eval_vec(pos, dd),
        }
    }
}

pub struct ErosionFbmBuilder {
    seed: u32,
    octaves: usize,
    lacunarity: f32,
    gain: f32,
    erosion_strength: f32,
    force_scalar: bool,
}

impl ErosionFbmBuilder {
    pub fn new() -> Self {
        ErosionFbmBuilder {
            seed: 0,
            octaves: 1,
            lacunarity: 2.0,
            gain: 0.5,
            erosion_strength: 1.0,
            force_scalar: false,
        }
    }

    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    pub fn octaves(mut self, octaves: usize) -> Self {
        self.octaves = octaves;
        self
    }

    pub fn lacunarity(mut self, lacunarity: f32) -> Self {
        self.lacunarity = lacunarity;
        self
    }

    pub fn gain(mut self, gain: f32) -> Self {
        self.gain = gain;
        self
    }

    pub fn erosion_strength(mut self, strength: f32) -> Self {
        self.erosion_strength = strength;
        self
    }

    pub fn force_scalar(mut self, force: bool) -> Self {
        self.force_scalar = force;
        self
    }

    pub fn build(self) -> ErosionFbmGenerator {
        #[cfg(target_arch = "x86_64")]
        {
            if !self.force_scalar && std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                let source = SimplexAvx::new(self.seed);
                return ErosionFbmGenerator::Avx(ErosionFbmAvx::new(source, self.octaves, self.lacunarity, self.gain, self.erosion_strength));
            }
        }
        
        let source = SimplexBuilder::new().seed(self.seed).force_scalar(self.force_scalar).build();
        ErosionFbmGenerator::Scalar(ErosionFbmVec::new(source, self.octaves, self.lacunarity, self.gain, self.erosion_strength))
    }
}
