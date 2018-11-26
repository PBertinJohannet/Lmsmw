//! # lmsmw
//!
//! Neural network training using levemberg marquardt algorithm with sherman morisson woodburry formula
//! Implemented entirely in Rust.
//!
//! ---
//!
//! Can also use basic gradient descent when the previous algorithm diverges/is to slow.
//!
//!
//!
//!
#![deny(missing_docs,
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    unused_import_braces,
    unused_qualifications)]


#[macro_use]
extern crate rulinalg;
extern crate rand;
extern crate num_traits;


#[macro_use]
pub mod macros;
pub mod network;
mod train;
mod netstruct;
mod sgd;
mod lvbm;
mod example;
pub mod lmsmw;


pub use crate::lmsmw::Learner;
pub use crate::lmsmw::ExamplesConfig;
pub use crate::example::Test;
pub use rulinalg::vector::Vector;
