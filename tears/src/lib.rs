#![feature(let_chains)]
#![feature(drain_filter)]
// #![feature(hash_raw_entry)]
// #![feature(get_many_mut)]
// #![feature(build_hasher_simple_hash_one)]

#[cfg(any(feature = "intel-mkl-system", feature = "intel-mkl-static"))]
extern crate intel_mkl_src as _src;

#[cfg(any(feature = "openblas-system", feature = "openblas-static"))]
extern crate openblas_src as _src;

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
// mod impls;
pub mod util_trait;
pub mod utils;

pub(crate) use datatype::match_datatype_arm;
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

#[cfg(feature = "lazy")]
pub use lazy::{
    flatten, get_partition_size, groupby, groupby_par, join_left, prepare_groupby, Context,
    DropNaMethod, Expr, ExprElement, ExprOut, ExprOutView, Exprs, JoinType, RefType,
    RollingTimeStartBy,
};

#[cfg(feature = "blas")]
pub use lazy::OlsResult;

pub use crate::core::{
    ArbArray, Arr, Arr1, Arr2, ArrBase, ArrBase1, ArrD, ArrOk, ArrView, ArrView1, ArrViewD,
    ArrViewMut, ArrViewMut1, ArrViewMutD, Dim1, WrapNdarray,
};
