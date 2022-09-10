use crate::datatype::Number;
use numpy::ndarray::{ArrayViewD, ArrayViewMutD, IxDyn};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::FromPyObject;

/// 只读的动态python array转换为确定维度的ArrayView或可变ArrayView
pub trait DynToArrayRead {
    type Type: Number;
    fn to_arrayd(&self) -> ArrayViewD<Self::Type>;
}

/// 读写的动态python array转换为确定维度的ArrayView或可变ArrayView
pub trait DynToArrayWrite {
    type Type: Number;
    fn to_arrayd(&mut self) -> ArrayViewMutD<Self::Type>;
}

impl<T: Number> DynToArrayRead for PyReadonlyArrayDyn<'_, T> {
    type Type = T;
    fn to_arrayd(&self) -> ArrayViewD<T> {
        self.as_array().into_dimensionality::<IxDyn>().unwrap()
    }
}

impl<T: Number> DynToArrayWrite for PyReadwriteArrayDyn<'_, T> {
    type Type = T;
    fn to_arrayd(&mut self) -> ArrayViewMutD<T> {
        self.as_array_mut().into_dimensionality::<IxDyn>().unwrap()
    }
}

// 所有可接受的python array类型
#[derive(FromPyObject)]
pub enum PyArrayOk<'py> {
    F32(&'py PyArrayDyn<f32>),
    F64(&'py PyArrayDyn<f64>),
    I32(&'py PyArrayDyn<i32>),
    I64(&'py PyArrayDyn<i64>),
    Usize(&'py PyArrayDyn<usize>),
}
