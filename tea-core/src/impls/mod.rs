#[cfg(feature = "method_1d")]
mod impl_1d_method;
#[cfg(feature = "method_1d")]
mod impl_basic;
mod impl_method;
#[cfg(feature = "time")]
mod impl_time;

mod impl_numeric;
mod impl_traits;

#[cfg(feature = "blas")]
mod linalg;
#[cfg(feature = "blas")]
pub use linalg::{conjugate, replicate, LeastSquaresResult};
