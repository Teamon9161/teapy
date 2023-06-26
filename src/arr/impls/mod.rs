mod impl_1d_method;
#[cfg(feature = "blas")]
mod linalg;
// #[cfg(feature = "blas")]
// mod impl_linalg;
mod impl_method;
mod impl_numeric;
mod impl_traits;

pub use linalg::{conjugate, replicate, LeastSquaresResult};
