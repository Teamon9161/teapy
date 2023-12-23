pub use crate::{
    arbarray::{ArbArray, ViewOnBase},
    arrok::ArrOk,
    match_all, match_arrok, ArrBase, Dim1,
};

pub use super::own::{arr0, Arr, Arr1, Arr2, ArrD};
pub use super::traits::WrapNdarray;
pub use super::view::{ArrView, ArrView1, ArrView2, ArrViewD};
pub use super::viewmut::{ArrViewMut, ArrViewMut1, ArrViewMut2, ArrViewMutD};
pub use datatype::{BoolType, Cast, DataType, GetDataType, GetNone, Number, OptUsize, PyValue};
pub use error::{StrError, TpResult};

#[cfg(feature = "blas")]
pub use super::impls::{conjugate, replicate, LeastSquaresResult};

#[cfg(feature = "time")]
pub use datatype::{DateTime, TimeDelta, TimeUnit};

#[cfg(feature = "method_1d")]
pub use super::iterators::{Iter, IterMut};
