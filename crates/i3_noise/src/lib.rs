#![allow(dead_code)]
#![allow(unused)]

// generator traits
mod scalar_generator;
pub use scalar_generator::{ScalarGenerator, ScalarPosition, ScalarSample};

mod vec_generator;
pub use vec_generator::{VecGenerator, VecPosition, VecSample};

// base noises
mod simplex;
pub use simplex::Simplex;

#[cfg(target_arch = "x86_64")]
mod simplex_avx;
#[cfg(target_arch = "x86_64")]
pub use simplex_avx::SimplexAvx;

mod builder;
pub use builder::{SimplexBuilder, SimplexGenerator, FbmBuilder, FbmGenerator, ErosionFbmBuilder, ErosionFbmGenerator};

// combinators
mod fbm;
pub use fbm::FbmVec;

#[cfg(target_arch = "x86_64")]
mod fbm_avx;
#[cfg(target_arch = "x86_64")]
pub use fbm_avx::FbmAvx;

mod erosion_fbm;
pub use erosion_fbm::ErosionFbmVec;

#[cfg(target_arch = "x86_64")]
mod erosion_fbm_avx;
#[cfg(target_arch = "x86_64")]
pub use erosion_fbm_avx::ErosionFbmAvx;
