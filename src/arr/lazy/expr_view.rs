use super::super::ArrViewD;
#[cfg(feature = "blas")]
use super::OlsResult;
use super::{ExprElement, ExprOut};
use std::sync::Arc;

// to get a view of ExprOut
#[derive(Debug, Clone)]
pub enum ExprOutView<'a, T> {
    Arr(ArrViewD<'a, T>),
    ArrVec(Vec<ArrViewD<'a, T>>),
    #[cfg(feature = "blas")]
    OlsRes(Arc<OlsResult<'a>>),
}

impl<'a, T> From<ArrViewD<'a, T>> for ExprOutView<'a, T> {
    fn from(view: ArrViewD<'a, T>) -> Self {
        ExprOutView::Arr(view)
    }
}

impl<'a, T> From<Vec<ArrViewD<'a, T>>> for ExprOutView<'a, T> {
    fn from(view: Vec<ArrViewD<'a, T>>) -> Self {
        ExprOutView::ArrVec(view)
    }
}

#[cfg(feature = "blas")]
impl<'a, T> From<Arc<OlsResult<'a>>> for ExprOutView<'a, T> {
    fn from(res: Arc<OlsResult<'a>>) -> Self {
        ExprOutView::OlsRes(res)
    }
}

impl<'a, T: ExprElement> From<ExprOutView<'a, T>> for ExprOut<'a, T> {
    fn from(view: ExprOutView<'a, T>) -> Self {
        match view {
            ExprOutView::Arr(arr) => arr.into(),
            ExprOutView::ArrVec(arr_vec) => arr_vec.into(),
            #[cfg(feature = "blas")]
            ExprOutView::OlsRes(res) => res.into(),
            // _ => unimplemented!("convert thid type of expression output to expression input is unimplemented")
        }
    }
}

impl<'a, T> ExprOutView<'a, T> {
    /// Cast the dtype of the expr output view without copy.
    ///
    /// # Safety
    ///
    /// The size of `T` and `T2` must be the same
    pub unsafe fn into_dtype<T2>(self) -> ExprOutView<'a, T2> {
        use std::mem;
        if mem::size_of::<T>() == mem::size_of::<T2>() {
            mem::transmute(self)
        } else {
            panic!("the size of new type is different when calling into_dtype for ExprOutView")
        }
    }

    pub fn try_into_arr(self) -> Result<ArrViewD<'a, T>, &'static str> {
        match self {
            ExprOutView::Arr(arr) => Ok(arr),
            _ => Err("The output of the expression is not an array"),
        }
    }

    pub fn try_into_arr_vec(self) -> Result<Vec<ArrViewD<'a, T>>, &'static str> {
        match self {
            ExprOutView::ArrVec(arr_vec) => Ok(arr_vec),
            _ => Err("The output of the expression is not a vector of array"),
        }
    }

    #[inline]
    pub fn into_arr(self) -> ArrViewD<'a, T> {
        self.try_into_arr().unwrap_or_else(|e| panic!("{}", e))
    }

    #[inline]
    pub fn into_arr_vec(self) -> Vec<ArrViewD<'a, T>> {
        self.try_into_arr_vec().unwrap_or_else(|e| panic!("{}", e))
    }

    pub fn try_as_arr<'b>(&'b self) -> Result<&'b ArrViewD<'a, T>, &'static str> {
        if let ExprOutView::Arr(arr) = self {
            Ok(arr)
        } else {
            Err("The output of the expression is not an array")
        }
    }

    pub fn try_as_arr_vec<'b>(&'b self) -> Result<&'b Vec<ArrViewD<'a, T>>, &'static str> {
        if let ExprOutView::ArrVec(arr_vec) = self {
            Ok(arr_vec)
        } else {
            Err("The output of the expression is not a vector of array")
        }
    }

    #[inline]
    pub fn as_arr<'b>(&'b self) -> &'b ArrViewD<'a, T> {
        self.try_as_arr().unwrap_or_else(|e| panic!("{}", e))
    }

    #[inline]
    pub fn as_arr_vec<'b>(&'b self) -> &'b Vec<ArrViewD<'a, T>> {
        self.try_as_arr_vec().unwrap_or_else(|e| panic!("{}", e))
    }
}
