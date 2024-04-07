#[cfg(feature = "lazy")]
#[macro_use]
extern crate tea_macros;

mod arrow_io;
mod colselect;

#[cfg(feature = "arw")]
pub use arrow_io::{read_ipc, read_ipc_schema, scan_ipc_lazy, DataDictIPCExt, ExprIPCExt};
pub use colselect::{ColSelect, SingleCol};
