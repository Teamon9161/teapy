// extern crate tea_core as core;
#[cfg(feature = "lazy")]
extern crate tea_lazy as lazy;

#[macro_use]
extern crate tea_macros;

#[macro_use]
mod macros;
#[cfg(feature = "agg")]
mod agg;
#[cfg(feature = "create")]
mod create;
#[cfg(feature = "map")]
mod map;

#[cfg(feature = "rolling")]
mod rolling;

#[cfg(feature = "agg")]
pub use agg::*;
#[cfg(feature = "map")]
pub use map::*;
#[cfg(feature = "rolling")]
pub use rolling::*;
