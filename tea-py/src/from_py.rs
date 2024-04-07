// #[cfg(feature = "lazy")]
// use crate::pylazy::PyDataDict;
#[cfg(feature = "lazy")]
use crate::pylazy::RefObj;
use ahash::HashMap;
use numpy::{
    datetime::{units, Datetime},
    PyArray, PyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::PyValueError, prelude::PyAnyMethods, Bound, FromPyObject, PyAny, PyObject,
    PyResult, Python, ToPyObject,
};
// use std::sync::Arc;
#[cfg(feature = "option_dtype")]
use tea_core::datatype::{OptF64, OptI64};
use tea_core::prelude::*;
#[cfg(feature = "arw")]
use tea_io::ColSelect;
#[cfg(feature = "lazy")]
use tea_lazy::Context;

#[derive(FromPyObject)]
pub enum Scalar {
    Bool(bool),
    F64(f64),
    I32(i32),
    Usize(usize),
    String(String),
    Object(PyValue),
}

macro_rules! impl_scalar {
    ($($arm: ident),*) => {
        impl ToPyObject for Scalar {
            fn to_object(&self, py: Python<'_>) -> PyObject {
                match self {
                    $(Scalar::$arm(e) => e.to_object(py)),*
                }
            }
        }
    };
}

impl_scalar!(Bool, F64, I32, Usize, String, Object);

// 所有可接受的python array类型
#[derive(FromPyObject)]
pub enum PyArrayOk<'py> {
    Bool(&'py PyArrayDyn<bool>),
    F32(&'py PyArrayDyn<f32>),
    F64(&'py PyArrayDyn<f64>),
    I32(&'py PyArrayDyn<i32>),
    I64(&'py PyArrayDyn<i64>),
    Usize(&'py PyArrayDyn<usize>),
    Object(&'py PyArrayDyn<PyValue>),
    DateTimeMs(&'py PyArrayDyn<Datetime<units::Milliseconds>>),
    DateTimeUs(&'py PyArrayDyn<Datetime<units::Microseconds>>),
    DateTimeNs(&'py PyArrayDyn<Datetime<units::Nanoseconds>>),
}

/// match the enum `PyArrayOk` to get the discrete dtype of `PyArray` so that we can
/// call functions on a `PyArray` of which dtype is known;
macro_rules! match_pyarray {

    ($pyarr: expr, $e: ident, $body: tt $(,$arm: ident)*) => {
        match $pyarr {
            $(PyArrayOk::$arm($e) => $body,)*
            _ => unimplemented!("match pyarray of this dtype is not implemented")
        }
    };

    ($pyarr: expr, $e: ident, $body: tt) => {
        match_pyarray!($pyarr, $e, $body, Bool, F32, F64, I32, I64, Usize, Object, DatetimeMs, DatetimeNs, DatetimeUs)
    };

    (numeric $pyarr: expr, $e: ident, $body: tt) => {
        match_pyarray!($pyarr, $e, $body, F32, F64, I32, I64, Usize)
    };

    (datetime $pyarr: expr, $e: ident, $body: tt) => {
        match_pyarray!($pyarr, $e, $body, DatetimeMs, DatetimeNs, DatetimeUs)
    };
}

impl<'py> PyArrayOk<'py> {
    pub fn is_object(&self) -> bool {
        matches!(self, PyArrayOk::Object(_))
    }

    pub fn is_datetime(&self) -> bool {
        use PyArrayOk::*;
        matches!(self, DateTimeMs(_) | DateTimeNs(_) | DateTimeUs(_))
    }

    pub fn into_object(self) -> PyResult<&'py PyArrayDyn<PyValue>> {
        if let PyArrayOk::Object(obj_arr) = self {
            Ok(obj_arr)
        } else {
            Err(PyValueError::new_err("Dtype of the array is not object"))
        }
    }

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
    // #[cfg(feature = "option_dtype")]
    // Boll(Vec<OptBool>)
    I64(Vec<i64>),
    #[cfg(feature = "option_dtype")]
    OptI64(Vec<OptI64>),
    F64(Vec<f64>),
    #[cfg(feature = "option_dtype")]
    OptF64(Vec<OptF64>),
    String(Vec<String>),
    Object(Vec<PyValue>),
}

macro_rules! match_pylist {
    ($list: expr, $l: ident, $body: tt) => {
        match $list {
            PyList::Bool($l) => $body,
            PyList::I64($l) => $body,
            #[cfg(feature = "option_dtype")]
            PyList::OptI64($l) => $body,
            PyList::F64($l) => $body,
            #[cfg(feature = "option_dtype")]
            PyList::OptF64($l) => $body,
            PyList::String($l) => $body,
            PyList::Object($l) => $body,
        }
    };
}

pub trait NoDim0 {
    fn no_dim0(self, py: Python) -> PyResult<PyObject>;
}

impl<T, D> NoDim0 for &PyArray<T, D> {
    fn no_dim0(self, py: Python) -> PyResult<PyObject> {
        if self.ndim() == 0 {
            Ok(self.call_method0("item")?.to_object(py))
        } else {
            Ok(self.to_object(py))
        }
    }
}

impl<T, D> NoDim0 for Bound<'_, PyArray<T, D>> {
    fn no_dim0(self, py: Python) -> PyResult<PyObject> {
        if self.ndim() == 0 {
            Ok(self.call_method0("item")?.to_object(py))
        } else {
            Ok(self.to_object(py))
        }
    }
}

#[cfg(feature = "lazy")]
#[derive(Clone, Default)]
pub struct PyContext<'py> {
    pub ct: Option<Context<'py>>,
    pub obj_map: HashMap<String, RefObj>,
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
    fn extract(ob: &'py PyAny) -> PyResult<PyContext<'py>> {
        if ob.is_none() {
            Ok(Self {
                ct: None,
                obj_map: Default::default(),
            })
        // } else if let Ok(dd) = ob.extract::<PyDataDict>() {
        //     Ok(Self {
        //         // safety: we can cast 'py to 'static as we are running in python
        //         ct: unsafe { Some(std::mem::transmute(Arc::new(dd.dd))) },
        //         obj_map: dd.obj_map,
        //     })
        // } else if ob.hasattr("_dd")? && ob.get_type().name()? == "DataDict" {
        //     let dd = ob.getattr("_dd")?.extract::<PyDataDict>()?;
        //     Ok(Self {
        //         ct: unsafe { Some(std::mem::transmute(Arc::new(dd.dd))) },
        //         obj_map: dd.obj_map,
        //     })
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
    fn extract(ob: &'py PyAny) -> PyResult<PyColSelect<'py>> {
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
