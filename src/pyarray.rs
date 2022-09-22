use numpy::PyArrayDyn;
use pyo3::FromPyObject;

// 所有可接受的python array类型
#[derive(FromPyObject)]
pub enum PyArrayOk<'py> {
    F32(&'py PyArrayDyn<f32>),
    F64(&'py PyArrayDyn<f64>),
    I32(&'py PyArrayDyn<i32>),
    I64(&'py PyArrayDyn<i64>),
    Usize(&'py PyArrayDyn<usize>),
}
