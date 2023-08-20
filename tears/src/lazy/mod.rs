pub mod expr;
#[macro_use]
pub mod exprs;
mod auto_impl_own;
mod expr_view;
mod groupby;
mod impl_cmp;
mod impl_mut;
mod impl_ops;
mod impl_own;
mod impl_view;
mod join;
#[cfg(feature = "blas")]
mod linalg;
mod unique;

pub use expr::{Expr, ExprElement, ExprOut, RefType};
pub use expr_view::ExprOutView;
pub use exprs::Exprs;
pub use groupby::{flatten, get_partition_size, groupby, groupby_par, prepare_groupby};
pub use impl_own::DropNaMethod;
pub use join::{join_left, JoinType};
#[cfg(feature = "blas")]
pub use linalg::OlsResult;
