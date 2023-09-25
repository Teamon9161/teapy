use std::fmt::Debug;

#[cfg(feature = "lazy")]
use crate::ExprElement;
use crate::{DataType, GetDataType, GetNone};
use numpy::{Element, PyArrayDescr};
use pyo3::{FromPyObject, PyAny, PyObject, PyResult, Python, ToPyObject};

#[derive(Clone)]
#[repr(transparent)]
pub struct PyValue(pub PyObject);

impl Debug for PyValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.0)
    }
}

impl GetNone for PyValue {
    fn none() -> Self {
        PyValue(Python::with_gil(|py| py.None()))
    }
    fn is_none(self) -> bool {
        Python::with_gil(|py| self.0.as_ref(py).is_none())
    }
}

impl PartialEq for PyValue {
    fn eq(&self, other: &Self) -> bool {
        Python::with_gil(|py| self.0.as_ref(py).eq(other.0.as_ref(py))).unwrap()
    }
}

// impl Serialize for PyValue {
//     fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error>
//     {
//         Python::with_gil(|py| {
//             let obj = self.0.as_ref(py);
//             serde_pickle::to_vec(&obj, Default::default())
//         })
//     }
// }

impl ToPyObject for PyValue {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        self.0.to_object(py)
    }
}

impl Default for PyValue {
    fn default() -> Self {
        PyValue(Python::with_gil(|py| py.None()))
    }
}

impl GetDataType for PyValue {
    // type Physical = PyObject;
    fn dtype() -> DataType {
        DataType::Object
    }
}

#[cfg(feature = "lazy")]
impl ExprElement for PyValue {}

unsafe impl Element for PyValue {
    const IS_COPY: bool = false;

    fn get_dtype(py: Python) -> &PyArrayDescr {
        PyArrayDescr::object(py)
    }
}

impl<'source> FromPyObject<'source> for PyValue {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        Ok(PyValue(ob.to_object(ob.py())))
    }
}
