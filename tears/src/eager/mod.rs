#[cfg(feature = "agg")]
mod agg;
#[cfg(feature = "arr_func")]
mod arr_func;
#[cfg(feature = "agg")]
mod corr;
mod impl_arrok;

#[cfg(feature = "agg")]
pub use agg::QuantileMethod;
#[cfg(feature = "arr_func")]
pub use arr_func::FillMethod;
#[cfg(all(feature = "agg", feature = "arr_func"))]
pub use arr_func::WinsorizeMethod;
#[cfg(feature = "agg")]
pub use corr::CorrMethod;
