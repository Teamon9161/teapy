// extern crate tea_core as core;
#[cfg(feature = "lazy")]
extern crate tea_lazy as lazy;

#[macro_use]
extern crate tea_macros;

mod from_py;
#[macro_use]
mod macros;
// #[cfg(feature = "blas")]
// pub mod linalg;
#[cfg(feature = "agg")]
pub mod agg;
#[cfg(feature = "create")]
pub mod create;
#[cfg(feature = "map")]
pub mod map;

#[cfg(feature = "rolling")]
pub mod rolling;

#[cfg(feature = "agg")]
pub use agg::*;
#[cfg(feature = "map")]
pub use map::*;
#[cfg(feature = "rolling")]
pub use rolling::*;
