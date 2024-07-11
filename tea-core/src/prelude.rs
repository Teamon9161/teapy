pub use crate::{
    arbarray::{ArbArray, ViewOnBase},
    arrok::ArrOk,
    match_arrok,
    py_dtype::Object,
    ArrBase, Dim1,
};

#[cfg(feature = "method_1d")]
pub use super::impls::BasicAggExt;
pub use super::own::{arr0, Arr, Arr1, Arr2, ArrD};
pub use super::traits::WrapNdarray;
pub use super::view::{ArrView, ArrView1, ArrView2, ArrViewD};
pub use super::viewmut::{ArrViewMut, ArrViewMut1, ArrViewMut2, ArrViewMutD};

#[cfg(feature = "blas")]
pub use super::impls::{conjugate, replicate, LeastSquaresResult};

#[cfg(feature = "method_1d")]
pub use super::iterators::{Iter, IterMut};

pub use tevec::prelude::*;
