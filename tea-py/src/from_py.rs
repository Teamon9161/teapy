// #[cfg(feature = "lazy")]
// use crate::pylazy::PyDataDict;
#[cfg(feature = "lazy")]
use crate::pylazy::RefObj;
use ahash::{HashMap, HashMapExt};
use numpy::{
    datetime::{units, Datetime},
    prelude::*,
    PyArray, PyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

#[cfg(feature = "arw")]
use tea_io::ColSelect;
#[cfg(feature = "lazy")]
use tea_lazy::Context;
use teapy_core::prelude::*;

#[derive(FromPyObject)]
pub enum Scalar {
    Bool(bool),
    F64(f64),
    I32(i32),
    Usize(usize),
    String(String),
    Object(Object),
}

macro_rules! impl_scalar {
    ($($arm: ident),*) => {
        // impl ToPyObject for Scalar {
        //     fn to_object(&self, py: Python<'_>) -> PyObject {
        //         match self {
        //             $(Scalar::$arm(e) => e.to_object(py)),*
        //         }
        //     }
        // }
        // impl IntoPyAny
    };
}

impl_scalar!(Bool, F64, I32, Usize, String, Object);

// 所有可接受的python array类型
#[derive(FromPyObject)]
pub enum PyArrayOk<'py> {
    Bool(Bound<'py, PyArrayDyn<bool>>),
    F32(Bound<'py, PyArrayDyn<f32>>),
    F64(Bound<'py, PyArrayDyn<f64>>),
    I32(Bound<'py, PyArrayDyn<i32>>),
    I64(Bound<'py, PyArrayDyn<i64>>),
    U64(Bound<'py, PyArrayDyn<u64>>),
    Usize(Bound<'py, PyArrayDyn<usize>>),
    Object(Bound<'py, PyArrayDyn<Object>>),
    DateTimeMs(Bound<'py, PyArrayDyn<Datetime<units::Milliseconds>>>),
    DateTimeUs(Bound<'py, PyArrayDyn<Datetime<units::Microseconds>>>),
    DateTimeNs(Bound<'py, PyArrayDyn<Datetime<units::Nanoseconds>>>),
}

/// match the enum `PyArrayOk` to get the discrete dtype of `PyArray` so that we can
/// call functions on a `PyArray` of which dtype is known;
macro_rules! match_pyarray {
    ($($tt: tt)*) => {
        $crate::teapy_core::match_enum!(PyArrayOk, $($tt)*)
    };
}

impl<'py> PyArrayOk<'py> {
    #[inline]
    pub fn is_object(&self) -> bool {
        matches!(self, PyArrayOk::Object(_))
    }

    #[inline]
    pub fn is_datetime(&self) -> bool {
        use PyArrayOk::*;
        matches!(self, DateTimeMs(_) | DateTimeNs(_) | DateTimeUs(_))
    }

    #[inline]
    pub fn into_object(self) -> PyResult<Bound<'py, PyArrayDyn<Object>>> {
        if let PyArrayOk::Object(obj_arr) = self {
            Ok(obj_arr)
        } else {
            Err(PyValueError::new_err("Dtype of the array is not object"))
        }
    }

    #[inline]
    pub fn object_to_string_arr(self) -> PyResult<ArrD<String>> {
        if let Ok(obj_arr) = self.into_object() {
            // let arr_readonly = obj_arr.readonly();
            // let obj_view = obj_arr.readonly().as_array();
            Ok(obj_arr.readonly().as_array().map(|v| v.to_string()).wrap())
        } else {
            Err(PyValueError::new_err("Dtype of the array is not object"))
        }
    }
}

// do not change the order of the variants
#[derive(FromPyObject)]
pub enum PyList {
    Bool(Vec<bool>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    String(Vec<String>),
    Object(Vec<Object>),
}

macro_rules! match_pylist {
    ($list: expr, $l: ident, $body: tt) => {
        match $list {
            PyList::Bool($l) => $body,
            PyList::I64($l) => $body,
            PyList::F64($l) => $body,
            PyList::String($l) => $body,
            PyList::Object($l) => $body,
        }
    };
}

pub trait NoDim0 {
    fn no_dim0(self, py: Python) -> PyResult<Py<PyAny>>;
}

impl<T, D> NoDim0 for &Bound<'_, PyArray<T, D>> {
    fn no_dim0(self, py: Python) -> PyResult<Py<PyAny>> {
        if self.ndim() == 0 {
            self.call_method0("item")?.into_py_any(py)
        } else {
            self.into_py_any(py)
        }
    }
}

impl<T, D> NoDim0 for Bound<'_, PyArray<T, D>> {
    fn no_dim0(self, py: Python) -> PyResult<Py<PyAny>> {
        if self.ndim() == 0 {
            self.call_method0("item")?.into_py_any(py)
        } else {
            self.into_py_any(py)
        }
    }
}

#[cfg(feature = "lazy")]
#[derive(Default)]
pub struct PyContext<'py> {
    pub ct: Option<Context<'py>>,
    pub obj_map: HashMap<String, RefObj>,
}

impl Clone for PyContext<'_> {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            ct: self.ct.clone(),
            obj_map: {
                let mut map = HashMap::with_capacity(self.obj_map.len());
                for (key, value) in self.obj_map.iter() {
                    map.insert(
                        key.clone(),
                        value
                            .as_ref()
                            .map(|v| v.iter().map(|o| o.clone_ref(py)).collect()),
                    );
                }
                map
            },
        })
    }
}

#[cfg(feature = "lazy")]
impl<'py> From<Context<'py>> for PyContext<'py> {
    fn from(ct: Context<'py>) -> Self {
        Self {
            ct: Some(ct),
            obj_map: Default::default(),
        }
    }
}

#[cfg(feature = "lazy")]
impl<'py> FromPyObject<'py> for PyContext<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<PyContext<'py>> {
        if ob.is_none() {
            Ok(Self {
                ct: None,
                obj_map: Default::default(),
            })
        } else {
            Err(PyValueError::new_err(
                "Cannot extract a Context from the object",
            ))
        }
    }
}

#[cfg(feature = "arw")]
pub struct PyColSelect<'py>(pub ColSelect<'py>);

#[cfg(feature = "arw")]
impl<'py> FromPyObject<'py> for PyColSelect<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<PyColSelect<'py>> {
        if ob.is_none() {
            Ok(Self(ColSelect::Null))
        } else if let Ok(idx) = ob.extract::<i32>() {
            Ok(Self(ColSelect::Idx(vec![idx])))
        } else if let Ok(name) = ob.extract::<String>() {
            Ok(Self(ColSelect::NameOwned(vec![name])))
        } else if let Ok(idx) = ob.extract::<Vec<i32>>() {
            Ok(Self(ColSelect::Idx(idx)))
        } else if let Ok(name) = ob.extract::<Vec<String>>() {
            Ok(Self(ColSelect::NameOwned(name)))
        } else {
            Err(PyValueError::new_err(
                "Cannot extract a ColSelect object from the given object",
            ))
        }
    }
}
