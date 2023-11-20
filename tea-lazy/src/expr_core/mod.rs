mod data;
mod expr;
mod expr_element;
mod expr_inner;
mod impls;

pub use data::Data;
pub use expr::Expr;
pub use expr_element::ExprElement;
pub use expr_inner::{FuncNode, FuncOut};

pub use impls::adjust_slice;
#[cfg(feature = "agg")]
pub use impls::corr;
#[cfg(all(feature = "arr_func", feature = "agg"))]
pub use impls::DropNaMethod;
#[cfg(feature = "window_func")]
pub use impls::RollingTimeStartBy;
