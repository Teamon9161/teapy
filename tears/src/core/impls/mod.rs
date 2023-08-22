mod impl_1d_method;
mod impl_method;
mod impl_numeric;
mod impl_traits;
#[cfg(feature = "blas")]
mod linalg;
#[cfg(feature = "blas")]
pub use linalg::{conjugate, replicate, LeastSquaresResult};
