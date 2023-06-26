use crate::arr::datatype::{DataType, GetDataType};
#[cfg(feature = "lazy")]
use crate::arr::{lazy::ExprElement, JoinType};
use numpy::{
    datetime::{units, Datetime},
    Element, PyArray, PyArray1, PyArrayDescr, PyArrayDyn,
};
use pyo3::{exceptions::PyValueError, FromPyObject, PyAny, PyObject, PyResult, Python, ToPyObject};

#[cfg(feature = "lazy")]
use crate::arr::DropNaMethod;
use crate::arr::{CorrMethod, FillMethod, QuantileMethod, WinsorizeMethod};

#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct PyValue(pub PyObject);

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
    type Physical = PyObject;
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
}

#[derive(FromPyObject)]
pub enum PyList {
    Bool(Vec<bool>),
    F64(Vec<f64>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    I32(Vec<i32>),
    String(Vec<String>),
    Object(Vec<PyValue>),
}

macro_rules! match_pylist {
    ($list: expr, $l: ident, $body: tt) => {
        match $list {
            PyList::Bool($l) => $body,
            PyList::I32($l) => $body,
            PyList::I64($l) => $body,
            PyList::F32($l) => $body,
            PyList::F64($l) => $body,
            PyList::String($l) => $body,
            PyList::Object($l) => $body,
        }
    };
}
// pub(crate) use match_pylist;

#[derive(FromPyObject)]
pub enum GroupByKey<'py> {
    // Str(&'py PyArray1<PyObject>),
    I32(&'py PyArray1<i32>),
    I64(&'py PyArray1<i64>),
    Usize(&'py PyArray1<usize>),
}

impl<'source> FromPyObject<'source> for CorrMethod {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("pearson").to_lowercase();
        let out = match s.as_str() {
            "pearson" => CorrMethod::Pearson,
            "spearman" => CorrMethod::Spearman,
            _ => panic!("Not supported method: {s} in correlation"),
        };
        Ok(out)
    }
}

impl<'source> FromPyObject<'source> for FillMethod {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("ffill").to_lowercase();
        let out = match s.as_str() {
            "ffill" => FillMethod::Ffill,
            "bfill" => FillMethod::Bfill,
            "vfill" => FillMethod::Vfill,
            _ => panic!("Not support method: {s} in fillna"),
        };
        Ok(out)
    }
}

impl<'source> FromPyObject<'source> for WinsorizeMethod {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("quantile").to_lowercase();
        let out = match s.as_str() {
            "quantile" => WinsorizeMethod::Quantile,
            "median" => WinsorizeMethod::Median,
            "sigma" => WinsorizeMethod::Sigma,
            _ => panic!("Not support {s} method in winsorize"),
        };
        Ok(out)
    }
}

impl<'source> FromPyObject<'source> for QuantileMethod {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("linear").to_lowercase();
        let out = match s.as_str() {
            "linear" => QuantileMethod::Linear,
            "lower" => QuantileMethod::Lower,
            "higher" => QuantileMethod::Higher,
            "midpoint" => QuantileMethod::MidPoint,
            _ => panic!("Not supported quantile method: {s}"),
        };
        Ok(out)
    }
}

#[cfg(feature = "lazy")]
impl<'source> FromPyObject<'source> for JoinType {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("left").to_lowercase();
        let out = match s.as_str() {
            "left" => JoinType::Left,
            "right" => JoinType::Right,
            "inner" => JoinType::Inner,
            "outer" => JoinType::Outer,
            _ => panic!("Not supported join method: {s}"),
        };
        Ok(out)
    }
}

#[cfg(feature = "lazy")]
impl<'source> FromPyObject<'source> for DropNaMethod {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("any").to_lowercase();
        let out = match s.as_str() {
            "all" => DropNaMethod::All,
            "any" => DropNaMethod::Any,
            _ => panic!("Not supported dropna method: {s}"),
        };
        Ok(out)
    }
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
