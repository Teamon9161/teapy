#![feature(drain_filter)] // need in DataDict drop_inplace

extern crate tea_core as core;

mod datadict;
pub mod expr_core;

#[cfg(feature = "blas")]
mod linalg;

pub use datadict::{ColumnSelector, Context, DataDict, GetMutOutput, GetOutput, SetInput};
pub use tea_hash::TpHashMap;

pub use expr_core::{adjust_slice, Data, Expr, ExprElement, FuncNode, FuncOut};
#[cfg(feature = "blas")]
pub use linalg::OlsResult;
