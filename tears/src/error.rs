use pyo3::{exceptions::PyValueError, prelude::PyErr};
use std::{borrow::Cow, fmt::Display};

pub type TpResult<T> = Result<T, StrError>;

#[derive(Debug)]
pub struct StrError(pub Cow<'static, str>);

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

impl StrError {
    pub fn to_py(self) -> PyErr {
        self.into()
    }

    // pub fn from_vec_res<T>(vec_res: Vec<TpResult<T>>) -> Option<Self> {
    //     let mut count = 0;
    //     let out = Self(Cow::Owned(
    //         vec_res
    //             .into_iter()
    //             .filter_map(|e| e.err())
    //             .map(|e| {
    //                 count += 1;
    //                 e.to_string()
    //             })
    //             .collect::<Vec<_>>()
    //             .join("\n"),
    //     ));
    //     if count > 0 {
    //         Some(out)
    //     } else {
    //         None
    //     }
    // }
}
