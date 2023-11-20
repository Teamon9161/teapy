#[cfg(feature = "method_1d")]
mod impl_1d_method;
mod impl_method;
#[cfg(feature = "method_1d")]
mod impl_basic_agg;

mod impl_numeric;
mod impl_traits;
#[cfg(feature = "blas")]
mod linalg;
#[cfg(feature = "blas")]
pub use linalg::{conjugate, replicate, LeastSquaresResult};
