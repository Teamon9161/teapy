#![feature(let_chains)]
#![feature(drain_filter)]

#[cfg(any(feature = "intel-mkl-system", feature = "intel-mkl-static"))]
extern crate intel_mkl_src as _src;

#[cfg(any(feature = "openblas-system", feature = "openblas-static"))]
extern crate openblas_src as _src;
#[macro_use]
mod core;
mod eager;
mod error;
mod export;
mod hash;
mod iterators;
mod macros;
#[macro_use]
mod from_py;

#[cfg(feature = "window_func")]
mod window;

pub mod datatype;
#[cfg(feature = "lazy")]
#[macro_use]
pub mod lazy;

#[cfg(feature = "arw")]
mod arrow_io;
// mod impls;
pub mod util_trait;
pub mod utils;

#[cfg(feature = "arw")]
pub use arrow_io::{read_ipc, ColSelect};
pub(crate) use datatype::match_datatype_arm;
// pub(crate) use crate::core::match_arbarray;
#[cfg(feature = "option_dtype")]
pub use datatype::{ArrToOpt, OptF32, OptF64, OptI32, OptI64};
pub use datatype::{
    BoolType, Cast, DataType, DateTime, GetDataType, GetNone, Number, OptUsize, PyValue, TimeDelta,
    TimeUnit,
};
pub use eager::{CorrMethod, FillMethod, QuantileMethod, WinsorizeMethod};
pub use error::{StrError, TpResult};

pub use iterators::{Iter, IterMut};
pub use util_trait::{CollectTrusted, CollectTrustedToVec, TrustedLen};
pub use utils::{kh_sum, EmptyNew};

pub use lazy::expr_core::{Data, DropNaMethod, Expr, ExprElement};
#[cfg(feature = "lazy")]
pub use lazy::Context;

#[cfg(feature = "blas")]
pub use lazy::OlsResult;

pub use crate::core::{
    ArbArray, Arr, Arr1, Arr2, ArrBase, ArrBase1, ArrD, ArrOk, ArrView, ArrView1, ArrView2,
    ArrViewD, ArrViewMut, ArrViewMut1, ArrViewMut2, ArrViewMutD, Dim1, ViewOnBase, WrapNdarray,
};
