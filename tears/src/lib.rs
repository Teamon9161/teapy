#![feature(let_chains)]
#![feature(drain_filter)]
#![feature(hash_raw_entry)]

#[cfg(any(feature = "intel-mkl-system", feature = "intel-mkl-static"))]
extern crate intel_mkl_src as _src;

#[cfg(any(feature = "openblas-system", feature = "openblas-static"))]
extern crate openblas_src as _src;

pub extern crate tea_dtype as datatype;
pub extern crate tea_error as error;
#[cfg(feature = "groupby")]
pub extern crate tea_hash as hash;
pub extern crate tea_utils as utils;

#[macro_use]
mod core;
mod arropt;
mod eager;
// mod error;
mod export;
// mod hash;
#[cfg(feature = "method_1d")]
mod iterators;
mod macros;
#[macro_use]
mod from_py;

// pub mod datatype;
#[cfg(feature = "lazy")]
#[macro_use]
pub mod lazy;

#[cfg(feature = "arw")]
mod arrow_io;
pub use arropt::ArrToOpt;
#[cfg(feature = "arw")]
pub use arrow_io::{read_ipc, ColSelect};

#[cfg(feature = "arr_func")]
pub use eager::FillMethod;
#[cfg(all(feature = "agg", feature = "arr_func"))]
pub use eager::WinsorizeMethod;
#[cfg(feature = "agg")]
pub use eager::{CorrMethod, QuantileMethod};
pub use error::{StrError, TpResult};
#[cfg(feature = "method_1d")]
pub use iterators::{Iter, IterMut};
pub use utils::{kh_sum, CollectTrusted, CollectTrustedToVec, TrustedLen};

#[cfg(all(feature = "lazy", feature = "arr_func", feature = "agg"))]
pub use lazy::expr_core::DropNaMethod;
#[cfg(feature = "lazy")]
pub use lazy::expr_core::{Data, Expr, ExprElement};
#[cfg(feature = "lazy")]
pub use lazy::Context;

#[cfg(feature = "blas")]
pub use lazy::OlsResult;

pub use crate::core::{
    ArbArray, Arr, Arr1, Arr2, ArrBase, ArrBase1, ArrD, ArrOk, ArrView, ArrView1, ArrView2,
    ArrViewD, ArrViewMut, ArrViewMut1, ArrViewMut2, ArrViewMutD, Dim1, ViewOnBase, WrapNdarray,
};
