use pyo3::{exceptions::PyValueError, prelude::PyErr};
use std::{borrow::Cow, error::Error, fmt::Display};

pub type TpResult<T> = Result<T, StrError>;

#[derive(Debug)]
pub struct StrError(pub Cow<'static, str>);

impl Error for StrError {}

impl Display for StrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.0)
    }
}

impl From<&'static str> for StrError {
    fn from(s: &'static str) -> Self {
        Self(Cow::Borrowed(s))
    }
}

impl From<String> for StrError {
    fn from(s: String) -> Self {
        Self(Cow::Owned(s))
    }
}

impl From<StrError> for PyErr {
    fn from(e: StrError) -> Self {
        PyValueError::new_err(e.0)
    }
}

impl<T> From<Vec<TpResult<T>>> for StrError {
    fn from(v: Vec<TpResult<T>>) -> Self {
        Self(Cow::Owned(
            v.into_iter()
                .filter_map(|e| e.err())
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("\n"),
        ))
    }
}

impl From<std::io::Error> for StrError {
    fn from(e: std::io::Error) -> Self {
        Self(Cow::Owned(e.to_string()))
    }
}

impl StrError {
    pub fn to_py(self) -> PyErr {
        self.into()
    }
}

#[cfg(feature = "arw")]
impl From<arrow::error::Error> for StrError {
    fn from(e: arrow::error::Error) -> Self {
        Self(Cow::Owned(e.to_string()))
    }
}
