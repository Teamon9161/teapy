pub use crate::{
    ArrBase, 
    Dim1,
    match_all, match_arrok,
    arrok::ArrOk, 
    arbarray::{ArbArray, ViewOnBase}, 
};

pub use super::own::{Arr, Arr1, Arr2, ArrD};
pub use super::view::{ArrView, ArrView1, ArrView2, ArrViewD};
pub use super::viewmut::{ArrViewMut, ArrViewMut1, ArrViewMut2, ArrViewMutD};
pub use super::traits::WrapNdarray;
pub use datatype::{Cast, GetDataType, DataType, GetNone, Number, BoolType, OptUsize};
pub use error::{TpResult, StrError};

#[cfg(feature = "blas")]
pub use super::impls::{conjugate, replicate, LeastSquaresResult};

#[cfg(feature = "time")]
pub use datatype::{DateTime, TimeDelta, TimeUnit};

#[cfg(feature = "method_1d")]
pub use super::iterators::{Iter, IterMut};