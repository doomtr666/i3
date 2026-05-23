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

// combinators
mod fbm;
pub use fbm::FbmVec;
