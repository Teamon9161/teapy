#[cfg(feature = "lazy")]
mod impl_lazy;
mod ipc;
pub(crate) mod utils;

#[cfg(feature = "lazy")]
pub use impl_lazy::{scan_ipc_lazy, DataDictIPCExt, ExprIPCExt};
pub use ipc::{read_ipc, read_ipc_schema};
