mod context;
mod datadict;
pub mod expr_core;
#[cfg(feature = "groupby")]
mod groupby;
#[cfg(feature = "groupby")]
mod join;
#[cfg(feature = "blas")]
mod linalg;
mod unique;

pub use context::Context;
pub use datadict::{ColumnSelector, DataDict, GetMutOutput, GetOutput, SetInput};
#[cfg(feature = "agg")]
pub use expr_core::corr;
#[cfg(all(feature = "arr_func", feature = "agg"))]
pub use expr_core::DropNaMethod;
pub use expr_core::{Data, Expr};
#[cfg(feature = "groupby")]
pub use groupby::{flatten, get_partition_size, groupby, groupby_par, prepare_groupby};
#[cfg(feature = "groupby")]
pub use join::{join_left, JoinType};

#[cfg(feature = "window_func")]
pub use expr_core::RollingTimeStartBy;

#[cfg(feature = "blas")]
pub use linalg::OlsResult;
