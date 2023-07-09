use pyo3::Python;

use crate::arr::{DateTime, TimeDelta, TimeUnit};
use crate::from_py::PyValue;

use super::super::{match_datatype_arm, DataType, GetDataType};
use super::expr_view::ExprOutView;
use super::{Expr, ExprElement, ExprInner};

#[derive(Clone)]
pub enum Exprs<'a> {
    F32(Expr<'a, f32>),
    F64(Expr<'a, f64>),
    I32(Expr<'a, i32>),
    I64(Expr<'a, i64>),
    Bool(Expr<'a, bool>),
    Usize(Expr<'a, usize>),
    Str(Expr<'a, &'a str>),
    String(Expr<'a, String>),
    Object(Expr<'a, PyValue>),
    DateTime(Expr<'a, DateTime>),
    TimeDelta(Expr<'a, TimeDelta>),
    OpUsize(Expr<'a, Option<usize>>),
}

#[macro_export]
macro_rules! match_ {
    // select the match arm
    ($enum: ident, $exprs: expr, $e: ident, $body: tt, $($arm: ident),*) => {
        match $exprs {
            $($enum::$arm($e) => $body,)*
            _ => unimplemented!("Not supported dtype in match_exprs")
        }
    };

    ($enum: ident, $exprs: expr, $e: ident, $body: tt) => {
        match_!($enum, $exprs, $e, $body, F32, F64, I32, I64, Bool, Usize, Str, String, Object, DateTime, TimeDelta, OpUsize)
    };

    ($enum: ident, ($exprs1: expr, $e1: ident, $($arm1: ident),*), ($exprs2: expr, $e2: ident, $($arm2: ident),*), $body: tt) => {
        match_!($enum, $exprs1, $e1, {match_!($enum, $exprs2, $e2, $body, $($arm2),*)}, $($arm1),*)
    };
}

#[macro_export]
macro_rules! match_exprs {
    (numeric $($tt: tt)*) => {match_!(Exprs, $($tt)*, F32, F64, I32, I64, Usize)};
    ($($tt: tt)*) => {match_!(Exprs, $($tt)*)};
}

macro_rules! match_exprs_inner {
    ($($tt: tt)*) => {match_!(ExprsInner, $($tt)*)}
}

impl<'a, T: ExprElement + 'a> From<Expr<'a, T>> for Exprs<'a> {
    fn from(expr: Expr<'a, T>) -> Self {
        unsafe {
            match T::dtype() {
                DataType::Bool => Exprs::Bool(expr.into_dtype::<bool>()),
                DataType::Usize => Exprs::Usize(expr.into_dtype::<usize>()),
                DataType::OpUsize => Exprs::OpUsize(expr.into_dtype::<Option<usize>>()),
                DataType::F32 => Exprs::F32(expr.into_dtype::<f32>()),
                DataType::F64 => Exprs::F64(expr.into_dtype::<f64>()),
                DataType::I32 => Exprs::I32(expr.into_dtype::<i32>()),
                DataType::I64 => Exprs::I64(expr.into_dtype::<i64>()),
                DataType::Str => Exprs::Str(expr.into_dtype::<&'a str>()),
                DataType::String => Exprs::String(expr.into_dtype::<String>()),
                DataType::Object => Exprs::Object(expr.into_dtype::<PyValue>()),
                DataType::DateTime => Exprs::DateTime(expr.into_dtype::<DateTime>()),
                DataType::TimeDelta => Exprs::TimeDelta(expr.into_dtype::<TimeDelta>()),
                // _ => {unimplemented!("create Expr of this dtype is not implemented")}
            }
        }
    }
}

impl<'a> Default for Exprs<'a> {
    fn default() -> Self {
        Exprs::I32(Expr::<'a, i32>::default())
    }
}

impl<'a> Exprs<'a> {
    #[allow(unreachable_patterns)]
    pub fn eval_inplace(&mut self) {
        match_exprs!(self, e, { e.eval_inplace() })
    }

    pub fn is_f32(&self) -> bool {
        matches!(self, Exprs::F32(_))
    }

    pub fn is_f64(&self) -> bool {
        matches!(self, Exprs::F64(_))
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Exprs::F64(_) | Exprs::F32(_))
    }

    pub fn is_i32(&self) -> bool {
        matches!(self, Exprs::I32(_))
    }

    pub fn is_i64(&self) -> bool {
        matches!(self, Exprs::I64(_))
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Exprs::I32(_) | Exprs::I64(_))
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, Exprs::Bool(_))
    }

    pub fn is_string(&self) -> bool {
        matches!(self, Exprs::String(_))
    }

    pub fn is_datetime(&self) -> bool {
        matches!(self, Exprs::DateTime(_))
    }

    pub fn is_timedelta(&self) -> bool {
        matches!(self, Exprs::TimeDelta(_))
    }

    pub fn is_str(&self) -> bool {
        matches!(self, Exprs::Str(_))
    }

    pub fn is_object(&self) -> bool {
        matches!(self, Exprs::Object(_))
    }

    pub fn cast_f64(self) -> Result<Expr<'a, f64>, &'static str> {
        match self {
            Exprs::F32(e) => Ok(e.cast::<f64>()),
            Exprs::F64(e) => Ok(e),
            Exprs::I32(e) => Ok(e.cast::<f64>()),
            Exprs::I64(e) => Ok(e.cast::<f64>()),
            Exprs::Bool(e) => Ok(e.cast::<i32>().cast::<f64>()),
            Exprs::Usize(e) => Ok(e.cast::<f64>()),
            _ => Err("Cast to f64 for this dtype is unimplemented"),
        }
    }

    pub fn cast_f32(self) -> Result<Expr<'a, f32>, &'static str> {
        match self {
            Exprs::F32(e) => Ok(e),
            Exprs::F64(e) => Ok(e.cast::<f32>()),
            Exprs::I32(e) => Ok(e.cast::<f32>()),
            Exprs::I64(e) => Ok(e.cast::<f32>()),
            Exprs::Bool(e) => Ok(e.cast::<i32>().cast::<f32>()),
            Exprs::Usize(e) => Ok(e.cast::<f32>()),
            _ => Err("Cast to f32 for this dtype is unimplemented"),
        }
    }

    pub fn cast_i32(self) -> Result<Expr<'a, i32>, &'static str> {
        match self {
            Exprs::F32(e) => Ok(e.cast::<i32>()),
            Exprs::F64(e) => Ok(e.cast::<i32>()),
            Exprs::I32(e) => Ok(e),
            Exprs::I64(e) => Ok(e.cast::<i32>()),
            Exprs::Bool(e) => Ok(e.cast::<i32>()),
            Exprs::Usize(e) => Ok(e.cast::<i32>()),
            // Exprs::DateTime(e) => {Ok(e.cast::<i32>())}
            _ => Err("Cast to i32 for this dtype is unimplemented"),
        }
    }

    pub fn cast_i64(self) -> Result<Expr<'a, i64>, &'static str> {
        match self {
            Exprs::F32(e) => Ok(e.cast::<i64>()),
            Exprs::F64(e) => Ok(e.cast::<i64>()),
            Exprs::I32(e) => Ok(e.cast::<i64>()),
            Exprs::I64(e) => Ok(e),
            Exprs::Bool(e) => Ok(e.cast::<i64>()),
            Exprs::Usize(e) => Ok(e.cast::<i64>()),
            _ => Err("Cast to i64 for this dtype is unimplemented"),
        }
    }

    pub fn cast_usize(self) -> Result<Expr<'a, usize>, &'static str> {
        match self {
            Exprs::F32(e) => Ok(e.cast::<usize>()),
            Exprs::F64(e) => Ok(e.cast::<usize>()),
            Exprs::I32(e) => Ok(e.cast::<usize>()),
            Exprs::I64(e) => Ok(e.cast::<usize>()),
            Exprs::Bool(e) => Ok(e.cast::<usize>()),
            Exprs::Usize(e) => Ok(e),
            _ => Err("Cast to usize for this dtype is unimplemented"),
        }
    }

    pub fn cast_object(self) -> Result<Expr<'a, PyValue>, &'static str> {
        match self {
            Exprs::Object(e) => Ok(e),
            _ => Err("Cast lazily to object for this dtype is unimplemented"),
        }
    }

    #[allow(unreachable_patterns)]
    pub fn cast_object_eager<'py>(
        self,
        py: Python<'py>,
    ) -> Result<Expr<'a, PyValue>, &'static str> {
        if self.is_object() {
            return self.cast_object();
        }
        match_exprs!(self, e, { Ok(e.cast_object_eager(py)) })
    }

    pub fn cast_str(self) -> Result<Expr<'a, &'a str>, &'static str> {
        match self {
            Exprs::Str(e) => Ok(e),
            _ => Err("Cast to str for this dtype is unimplemented"),
        }
    }

    pub fn cast_string(self) -> Result<Expr<'a, String>, &'static str> {
        match self {
            Exprs::String(e) => Ok(e),
            Exprs::Str(e) => Ok(e.cast_string()),
            Exprs::F32(e) => Ok(e.cast_string()),
            Exprs::F64(e) => Ok(e.cast_string()),
            Exprs::I32(e) => Ok(e.cast_string()),
            Exprs::I64(e) => Ok(e.cast_string()),
            Exprs::DateTime(e) => Ok(e.strftime(None)),
            _ => Err("Cast to string for this dtype is unimplemented"),
        }
    }

    pub fn cast_bool(self) -> Result<Expr<'a, bool>, &'static str> {
        match self {
            Exprs::Bool(e) => Ok(e),
            Exprs::F32(e) => Ok(e.cast_bool()),
            Exprs::F64(e) => Ok(e.cast_bool()),
            Exprs::I32(e) => Ok(e.cast_bool()),
            Exprs::I64(e) => Ok(e.cast_bool()),
            _ => Err("Cast to bool for this dtype is unimplemented"),
        }
    }

    pub fn cast_datetime(self, unit: Option<TimeUnit>) -> Result<Expr<'a, DateTime>, &'static str> {
        match self {
            Exprs::DateTime(e) => Ok(e),
            Exprs::I32(e) => Ok(e.cast_datetime(unit)),
            Exprs::I64(e) => Ok(e.cast_datetime(unit)),
            Exprs::F64(e) => Ok(e.cast_datetime(unit)),
            Exprs::F32(e) => Ok(e.cast_datetime(unit)),
            Exprs::Usize(e) => Ok(e.cast_datetime(unit)),
            _ => Err("Cast lazily to datetime for this dtype is unimplemented"),
        }
    }

    pub fn cast_datetime_default(self) -> Result<Expr<'a, DateTime>, &'static str> {
        self.cast_datetime(Some(Default::default()))
    }

    pub fn cast_timedelta(self) -> Result<Expr<'a, TimeDelta>, &'static str> {
        match self {
            Exprs::TimeDelta(e) => Ok(e),
            Exprs::String(e) => Ok(e.cast_timedelta()),
            _ => Err("Cast to timedelta for this dtype is unimplemented"),
        }
    }
}

#[derive(Debug)]
pub(super) enum ExprsInner<'a> {
    F32(ExprInner<'a, f32>),
    F64(ExprInner<'a, f64>),
    I32(ExprInner<'a, i32>),
    I64(ExprInner<'a, i64>),
    Bool(ExprInner<'a, bool>),
    Usize(ExprInner<'a, usize>),
    Str(ExprInner<'a, &'a str>),
    String(ExprInner<'a, String>),
    DateTime(ExprInner<'a, DateTime>),
    TimeDelta(ExprInner<'a, TimeDelta>),
    Object(ExprInner<'a, PyValue>),
    OpUsize(ExprInner<'a, Option<usize>>),
}

impl<'a, T: ExprElement> From<ExprInner<'a, T>> for ExprsInner<'a> {
    fn from(expr: ExprInner<'a, T>) -> Self {
        unsafe {
            match T::dtype() {
                DataType::Bool => ExprsInner::Bool(expr.into_dtype::<bool>()),
                DataType::Usize => ExprsInner::Usize(expr.into_dtype::<usize>()),
                DataType::OpUsize => ExprsInner::OpUsize(expr.into_dtype::<Option<usize>>()),
                DataType::F32 => ExprsInner::F32(expr.into_dtype::<f32>()),
                DataType::F64 => ExprsInner::F64(expr.into_dtype::<f64>()),
                DataType::I32 => ExprsInner::I32(expr.into_dtype::<i32>()),
                DataType::I64 => ExprsInner::I64(expr.into_dtype::<i64>()),
                DataType::Str => ExprsInner::Str(expr.into_dtype::<&'a str>()),
                DataType::String => ExprsInner::String(expr.into_dtype::<String>()),
                DataType::Object => ExprsInner::Object(expr.into_dtype::<PyValue>()),
                DataType::DateTime => ExprsInner::DateTime(expr.into_dtype::<DateTime>()),
                DataType::TimeDelta => ExprsInner::TimeDelta(expr.into_dtype::<TimeDelta>()),
                // _ => {unimplemented!("create Expr of this dtype is not implemented")}
            }
        }
    }
}

impl Default for ExprsInner<'_> {
    fn default() -> Self {
        ExprsInner::I32(ExprInner::<i32>::default())
    }
}

impl<'a> ExprsInner<'a> {
    /// Match an arm when we have known the dtype. Consume Exprs
    ///
    /// # Safety
    ///
    /// `T::dtype()` must match the arm of `Exprs`
    #[allow(unreachable_patterns)]
    pub(super) unsafe fn downcast<T: ExprElement>(self) -> ExprInner<'a, T> {
        match_datatype_arm!(
            self,
            e,
            ExprsInner,
            T,
            (Bool, F32, F64, I32, I64, Usize, Str, String, Object, DateTime, TimeDelta, OpUsize),
            { e.into_dtype::<T>() }
        )
    }

    /// Match an arm when we have known the dtype.
    ///
    /// # Safety
    ///
    /// `T::dtype()` must match the arm of `Exprs`
    #[allow(unreachable_patterns)]
    pub(super) unsafe fn get<T: ExprElement>(&self) -> &ExprInner<'a, T> {
        match_datatype_arm!(
            self,
            e,
            ExprsInner,
            T,
            (Bool, F32, F64, I32, I64, Usize, Str, String, Object, DateTime, TimeDelta, OpUsize),
            { std::mem::transmute(e) }
        )
    }

    /// Match an arm when we have known the dtype.
    ///
    /// # Safety
    ///
    /// `T::dtype()` must match the arm of `Exprs`
    #[allow(unreachable_patterns)]
    pub(super) unsafe fn get_mut<T: ExprElement>(&mut self) -> &mut ExprInner<'a, T> {
        match_datatype_arm!(
            self,
            e,
            ExprsInner,
            T,
            (Bool, F32, F64, I32, I64, Usize, Str, String, Object, DateTime, TimeDelta, OpUsize),
            { std::mem::transmute(e) }
        )
    }

    #[allow(unreachable_patterns)]
    pub(super) fn step(&self) -> usize {
        match_exprs_inner!(self, e, { e.step() })
    }

    #[allow(dead_code)]
    #[allow(unreachable_patterns)]
    pub(super) fn get_owned(&self) -> Option<bool> {
        match_exprs_inner!(self, e, { e.get_owned() })
    }

    #[allow(unreachable_patterns)]
    pub(super) fn step_acc(&self) -> usize {
        match_exprs_inner!(self, e, { e.step_acc() })
    }

    // #[allow(unreachable_patterns)]
    // pub(super) fn view_arr<T: GetDataType>(&self) -> Result<ArrViewD<'_, T>, &'static str> {
    //     // we have known the datatype of the enum ,so only one arm will be executed
    //     match_datatype_arm!(
    //         self,
    //         e,
    //         ExprsInner,
    //         T,
    //         (Bool, F32, F64, I32, I64, Usize, Str, String, Object, DateTime, TimeDelta, OpUsize),
    //         { e.try_view_arr().map(|arr| unsafe { arr.into_dtype::<T>() }) }
    //     )
    // }

    #[allow(unreachable_patterns)]
    pub(super) fn view<T: GetDataType>(&self) -> Result<ExprOutView<'_, T>, &'static str> {
        // we have known the datatype of the enum ,so only one arm will be executed
        match_datatype_arm!(
            self,
            e,
            ExprsInner,
            T,
            (Bool, F32, F64, I32, I64, Usize, Str, String, Object, DateTime, TimeDelta, OpUsize),
            { e.try_view().map(|view| unsafe { view.into_dtype::<T>() }) }
        )
    }

    #[allow(unreachable_patterns)]
    pub(super) fn eval_inplace(&mut self) {
        match_exprs_inner!(self, e, { e.eval_inplace() })
    }
}
