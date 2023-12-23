use super::ArrBase;
use ndarray::{ArrayBase, Dimension, RawData, RawDataClone};
use std::convert::From;

pub trait WrapNdarray<S: RawData, D: Dimension> {
    fn wrap(self) -> ArrBase<S, D>;
}

impl<S: RawData, D: Dimension> WrapNdarray<S, D> for ArrayBase<S, D> {
    #[inline(always)]
    fn wrap(self) -> ArrBase<S, D> {
        ArrBase::new(self)
    }
}

impl<S: RawData, D: Dimension> From<ArrayBase<S, D>> for ArrBase<S, D> {
    #[inline(always)]
    fn from(arr: ArrayBase<S, D>) -> Self {
        ArrBase::new(arr)
    }
}

impl<S: RawDataClone, D: Clone + Dimension> Clone for ArrBase<S, D> {
    #[inline(always)]
    fn clone(&self) -> Self {
        self.0.clone().wrap()
    }
}
