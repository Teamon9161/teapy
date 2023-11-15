mod algos;
mod alloc;
#[macro_use]
mod macros;

pub use algos::{fast2sum, kh_sum, vec_fold, vec_nfold};
pub use alloc::{vec_uninit, VecAssumeInit};
