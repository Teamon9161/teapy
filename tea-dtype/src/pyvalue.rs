use std::fmt::Debug;

use super::cast::Cast;
use crate::{DataType, GetDataType, GetNone};
use numpy::{Element, PyArrayDescr};
use pyo3::{Bound, FromPyObject, PyAny, PyObject, PyResult, Python, ToPyObject};
#[cfg(feature = "serde")]
use serde::{Serialize, Serializer};
use std::string::ToString;

#[derive(Clone)]
#[repr(transparent)]
pub struct PyValue(pub PyObject);

impl ToString for PyValue {
    #[inline(always)]
    fn to_string(&self) -> String {
        self.0.to_string()
    }
}

impl Debug for PyValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.0)
    }
}

impl GetNone for PyValue {
    #[inline(always)]
    fn none() -> Self {
        PyValue(Python::with_gil(|py| py.None()))
    }

    #[inline(always)]
    fn is_none(&self) -> bool {
        Python::with_gil(|py| self.0.as_ref(py).is_none())
    }
}

impl PartialEq for PyValue {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Python::with_gil(|py| self.0.as_ref(py).eq(other.0.as_ref(py))).unwrap()
    }
}

#[cfg(feature = "serde")]
impl Serialize for PyValue {
    fn serialize<S: Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
        unimplemented!("can not serialize PyObject")
    }
}

impl ToPyObject for PyValue {
    #[inline(always)]
    fn to_object(&self, py: Python<'_>) -> PyObject {
        self.0.to_object(py)
    }
}

impl Default for PyValue {
    #[inline(always)]
    fn default() -> Self {
        PyValue(Python::with_gil(|py| py.None()))
    }
}

impl GetDataType for PyValue {
    // type Physical = PyObject;
    #[inline(always)]
    fn dtype() -> DataType {
        DataType::Object
    }
}

// #[cfg(feature = "lazy")]
// impl ExprElement for PyValue {}

unsafe impl Element for PyValue {
    const IS_COPY: bool = false;
    // #[inline(always)]
    // fn get_dtype(py: Python) -> &PyArrayDescr {
    //     PyArrayDescr::object(py)
    // }

    #[inline(always)]
    fn get_dtype_bound(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        PyArrayDescr::object_bound(py)
    }
}

impl<'source> FromPyObject<'source> for PyValue {
    #[inline(always)]
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        Ok(PyValue(ob.to_object(ob.py())))
    }
}

// impl Cast<String> for PyValue {
//     #[inline]
//     fn cast(self) -> String {
//         self.to_string()
//     }
// }

impl<T> Cast<T> for PyValue
where
    String: Cast<T>,
{
    #[inline]
    fn cast(self) -> T {
        self.0.to_string().cast()
    }
}
