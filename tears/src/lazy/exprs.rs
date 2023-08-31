// use parking_lot::Mutex;
use pyo3::Python;
use std::fmt::Debug;
// use std::sync::Arc;

#[cfg(feature = "option_dtype")]
use crate::datatype::{OptF32, OptF64, OptI32, OptI64};
use crate::{Cast, DateTime, OptUsize, PyValue, TimeDelta, TimeUnit, TpResult};

use super::super::{match_datatype_arm, DataType, GetDataType};
use super::context::Context;
use super::expr::{Expr, ExprElement, ExprInner};
use super::expr_view::ExprOutView;
// use super::ColumnSelector;

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
    VecUsize(Expr<'a, Vec<usize>>),
    // OpUsize(Expr<'a, Option<usize>>),
    #[cfg(feature = "option_dtype")]
    OptF32(Expr<'a, OptF32>),
    #[cfg(feature = "option_dtype")]
    OptF64(Expr<'a, OptF64>),
    #[cfg(feature = "option_dtype")]
    OptI32(Expr<'a, OptI32>),
    #[cfg(feature = "option_dtype")]
    OptI64(Expr<'a, OptI64>),
    OptUsize(Expr<'a, OptUsize>),
}

#[macro_export]
macro_rules! match_ {
    // select the match arm
    ($enum: ident, $exprs: expr, $e: ident, $body: tt, $($(#[$meta: meta])? $arm: ident),*) => {
        match $exprs {
            $($(#[$meta])? $enum::$arm($e) => $body,)*
            _ => unimplemented!("Not supported dtype in match_exprs")
        }
    };

    ($enum: ident, $exprs: expr, $e: ident, $body: tt) => {
        {
            #[cfg(feature="option_dtype")]
            macro_rules! inner_macro {
                () => {
                    match_!($enum, $exprs, $e, $body, F32, F64, I32, I64, Bool, Usize, Str, String, Object, DateTime, TimeDelta, VecUsize, OptF32, OptF64, OptI32, OptI64, OptUsize)
                };
            }

            #[cfg(not(feature="option_dtype"))]
            macro_rules! inner_macro {
                () => {
                    match_!($enum, $exprs, $e, $body, F32, F64, I32, I64, Bool, Usize, Str, String, Object, DateTime, TimeDelta, OptUsize, VecUsize)
                };
            }
            inner_macro!()
        }

    };

    ($enum: ident, ($exprs1: expr, $e1: ident, $($arm1: ident),*), ($exprs2: expr, $e2: ident, $($arm2: ident),*), $body: tt) => {
        match_!($enum, $exprs1, $e1, {match_!($enum, $exprs2, $e2, $body, $($arm2),*)}, $($arm1),*)
    };
}

#[macro_export]
macro_rules! match_exprs {
    (numeric $($tt: tt)*) => {match_!(Exprs, $($tt)*, F32, F64, I32, I64, Usize)};
    (hash $($tt: tt)*) => {match_!(Exprs, $($tt)*, F32, F64, I32, I64, Usize, String, Str, DateTime)};
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
                DataType::F32 => Exprs::F32(expr.into_dtype::<f32>()),
                DataType::F64 => Exprs::F64(expr.into_dtype::<f64>()),
                DataType::I32 => Exprs::I32(expr.into_dtype::<i32>()),
                DataType::I64 => Exprs::I64(expr.into_dtype::<i64>()),
                DataType::Str => Exprs::Str(expr.into_dtype::<&'a str>()),
                DataType::String => Exprs::String(expr.into_dtype::<String>()),
                DataType::Object => Exprs::Object(expr.into_dtype::<PyValue>()),
                DataType::DateTime => Exprs::DateTime(expr.into_dtype::<DateTime>()),
                DataType::TimeDelta => Exprs::TimeDelta(expr.into_dtype::<TimeDelta>()),
                DataType::VecUsize => Exprs::VecUsize(expr.into_dtype::<Vec<usize>>()),
                #[cfg(feature = "option_dtype")]
                DataType::OptF64 => Exprs::OptF64(expr.into_dtype::<OptF64>()),
                #[cfg(feature = "option_dtype")]
                DataType::OptF32 => Exprs::OptF32(expr.into_dtype::<OptF32>()),
                #[cfg(feature = "option_dtype")]
                DataType::OptI32 => Exprs::OptI32(expr.into_dtype::<OptI32>()),
                #[cfg(feature = "option_dtype")]
                DataType::OptI64 => Exprs::OptI64(expr.into_dtype::<OptI64>()),
                DataType::OptUsize => Exprs::OptUsize(expr.into_dtype::<OptUsize>()),
                DataType::U64 => unreachable!("U64 is not supported"),
            }
        }
    }
}

impl<'a> Default for Exprs<'a> {
    fn default() -> Self {
        Exprs::I32(Expr::<'a, i32>::default())
    }
}

impl Debug for Exprs<'_> {
    #[allow(unreachable_patterns)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match_exprs!(self, e, { e.fmt(f) })
    }
}

impl<'a> Exprs<'a> {
    #[allow(unreachable_patterns)]
    pub fn name(&self) -> Option<String> {
        match_exprs!(self, e, { e.name() })
    }

    #[allow(unreachable_patterns)]
    pub fn is_owned(&self) -> Option<bool> {
        match_exprs!(self, e, { e.is_owned() })
    }

    #[allow(unreachable_patterns)]
    pub fn is_context(&self) -> bool {
        match_exprs!(self, e, { e.is_context() })
    }

    // #[allow(unreachable_patterns)]
    // pub(super) fn into_inner(self) -> ExprsInner<'a> {
    //     match_exprs!(self, e, { e.downcast().into() })
    // }

    // #[allow(unreachable_patterns)]
    // pub fn cast_by_context(&mut self, context: Option<Context<'a>>) -> TpResult<&mut Self> {
    //     if let Some(context) = context {
    //         let exprs = std::mem::take(self);
    //         *self = match_exprs!(exprs, e, { e.cast_by_context(Some(context))?.into() });
    //     }
    //     Ok(self)
    // }

    /// # Safety
    ///
    /// the name should not be changed by another thread
    #[allow(unreachable_patterns)]
    pub unsafe fn ref_name(&self) -> Option<&str> {
        match_exprs!(self, e, { e.ref_name() })
    }

    pub fn dtype(&self) -> &'static str {
        use Exprs::*;
        match &self {
            F32(_) => "Float32",
            F64(_) => "Float64",
            I32(_) => "Int32",
            I64(_) => "Int64",
            Usize(_) => "Usize",
            Bool(_) => "Bool",
            String(_) => "String",
            Str(_) => "Str",
            Object(_) => "Object",
            DateTime(_) => "DateTime",
            TimeDelta(_) => "TimeDelta",
            OptUsize(_) => "Option<Usize>",
            VecUsize(_) => "Vec<Usize>",
            #[cfg(feature = "option_dtype")]
            OptF64(_) => "Option<F64>",
            #[cfg(feature = "option_dtype")]
            OptF32(_) => "Option<F32>",
            #[cfg(feature = "option_dtype")]
            OptI32(_) => "Option<I32>",
            #[cfg(feature = "option_dtype")]
            OptI64(_) => "Option<I64>",
        }
    }

    #[allow(unreachable_patterns)]
    pub fn rename(&mut self, name: String) {
        match_exprs!(self, e, { e.rename(name) })
    }

    #[allow(unreachable_patterns)]
    pub fn eval_inplace(&mut self, context: Option<Context<'a>>) -> TpResult<Option<Context<'a>>> {
        match_exprs!(self, e, { e.eval_inplace(context) })
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

    pub fn cast_f64(self) -> TpResult<Expr<'a, f64>> {
        Ok(self.cast())
    }

    #[cfg(feature = "option_dtype")]
    pub fn cast_optf64(self) -> TpResult<Expr<'a, OptF64>> {
        Ok(self.cast())
    }

    #[cfg(feature = "option_dtype")]
    pub fn cast_optf32(self) -> TpResult<Expr<'a, OptF32>> {
        Ok(self.cast())
    }

    #[cfg(feature = "option_dtype")]
    pub fn cast_opti32(self) -> TpResult<Expr<'a, OptI32>> {
        Ok(self.cast())
    }

    #[cfg(feature = "option_dtype")]
    pub fn cast_opti64(self) -> TpResult<Expr<'a, OptI64>> {
        Ok(self.cast())
    }

    // #[cfg(feature = "option_dtype")]
    pub fn cast_optusize(self) -> TpResult<Expr<'a, OptUsize>> {
        Ok(self.cast())
    }

    pub fn cast_f32(self) -> TpResult<Expr<'a, f32>> {
        Ok(self.cast())
    }

    pub fn cast_i32(self) -> TpResult<Expr<'a, i32>> {
        Ok(self.cast())
    }

    pub fn cast_i64(self) -> TpResult<Expr<'a, i64>> {
        Ok(self.cast())
    }

    pub fn cast_usize(self) -> TpResult<Expr<'a, usize>> {
        Ok(self.cast())
    }

    pub fn cast_object(self) -> TpResult<Expr<'a, PyValue>> {
        match self {
            Exprs::Object(e) => Ok(e),
            _ => Err("Cast lazily to object for this dtype is unimplemented".into()),
        }
    }

    pub fn cast_vecusize(self) -> TpResult<Expr<'a, Vec<usize>>> {
        match self {
            Exprs::VecUsize(e) => Ok(e),
            _ => Err("Cast lazily to object for this dtype is unimplemented".into()),
        }
    }

    #[allow(unreachable_patterns)]
    pub fn cast_object_eager<'py>(self, py: Python<'py>) -> TpResult<Expr<'a, PyValue>> {
        if self.is_object() {
            return self.cast_object();
        }
        match_exprs!(self, e, { e.cast_object_eager(py) })
    }

    pub fn cast_str(self) -> TpResult<Expr<'a, &'a str>> {
        match self {
            Exprs::Str(e) => Ok(e),
            _ => Err("Cast to str for this dtype is unimplemented".into()),
        }
    }

    pub fn cast_string(self) -> TpResult<Expr<'a, String>> {
        Ok(self.cast())
    }

    pub fn cast_bool(self) -> TpResult<Expr<'a, bool>> {
        Ok(self.cast())
    }

    pub fn cast_datetime(self, unit: Option<TimeUnit>) -> TpResult<Expr<'a, DateTime>> {
        match self {
            Exprs::DateTime(e) => Ok(e),
            Exprs::I32(e) => Ok(e.cast_datetime(unit)),
            Exprs::I64(e) => Ok(e.cast_datetime(unit)),
            Exprs::F64(e) => Ok(e.cast_datetime(unit)),
            Exprs::F32(e) => Ok(e.cast_datetime(unit)),
            Exprs::Usize(e) => Ok(e.cast_datetime(unit)),
            _ => Err("Cast lazily to datetime for this dtype is unimplemented".into()),
        }
    }
    pub fn cast_datetime_default(self) -> TpResult<Expr<'a, DateTime>> {
        Ok(self.cast())
    }

    pub fn cast_timedelta(self) -> TpResult<Expr<'a, TimeDelta>> {
        match self {
            Exprs::TimeDelta(e) => Ok(e),
            Exprs::String(e) => Ok(e.cast_timedelta()),
            _ => Err("Cast to timedelta for this dtype is unimplemented".into()),
        }
    }
}

macro_rules! impl_expr_cast {
    ($($(#[$meta: meta])? $T: ty),*) => {
        $(
            $(#[$meta])?
            impl<'a> Cast<Expr<'a, $T>> for Exprs<'a>
            {
                fn cast(self) -> Expr<'a, $T> {
                    match self {
                        Exprs::F32(e) => e.cast::<$T>(),
                        Exprs::F64(e) => e.cast::<$T>(),
                        Exprs::I32(e) => e.cast::<$T>(),
                        Exprs::I64(e) => e.cast::<$T>(),
                        Exprs::Bool(e) => e.cast::<i32>().cast::<$T>(),
                        Exprs::Usize(e) => e.cast::<$T>(),
                        // Exprs::Str(e) => e.cast::<$T>(),
                        Exprs::String(e) => e.cast::<$T>(),
                        // Exprs::Object(e) => e.cast::<$T>(),
                        Exprs::DateTime(e) => e.cast::<i64>().cast::<$T>(),
                        Exprs::TimeDelta(e) => e.cast::<i64>().cast::<$T>(),
                        #[cfg(feature = "option_dtype")]
                        Exprs::OptF64(e) => e.cast::<$T>(),
                        #[cfg(feature = "option_dtype")]
                        Exprs::OptF32(e) => e.cast::<$T>(),
                        #[cfg(feature = "option_dtype")]
                        Exprs::OptI32(e) => e.cast::<$T>(),
                        #[cfg(feature = "option_dtype")]
                        Exprs::OptI64(e) => e.cast::<$T>(),
                        Exprs::OptUsize(e) => e.cast::<$T>(),
                        _ => unimplemented!("Cast to this dtype is unimplemented"),
                    }
                }
            }
        )*
    };
}
impl_expr_cast!(
    i32,
    i64,
    f32,
    f64,
    usize,
    bool,
    String,
    DateTime,
    OptUsize,
    TimeDelta,
    #[cfg(feature = "option_dtype")]
    OptF32,
    #[cfg(feature = "option_dtype")]
    OptF64,
    #[cfg(feature = "option_dtype")]
    OptI32,
    #[cfg(feature = "option_dtype")]
    OptI64
);

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
    OptUsize(ExprInner<'a, OptUsize>),
    VecUsize(ExprInner<'a, Vec<usize>>),
    #[cfg(feature = "option_dtype")]
    OptF64(ExprInner<'a, OptF64>),
    #[cfg(feature = "option_dtype")]
    OptF32(ExprInner<'a, OptF32>),
    #[cfg(feature = "option_dtype")]
    OptI32(ExprInner<'a, OptI32>),
    #[cfg(feature = "option_dtype")]
    OptI64(ExprInner<'a, OptI64>),
    // #[cfg(feature = "option_dtype")]
    // OptUsize(ExprInner<'a, OptUsize>),
}

impl<'a> Debug for ExprsInner<'a> {
    #[allow(unreachable_patterns)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match_exprs_inner!(self, e, { e.fmt(f) })
    }
}

macro_rules! impl_from_trait {
    ($($(#[$meta: meta])? $ty: ident: $real: ty),*) => {
        impl<'a, T: ExprElement> From<ExprInner<'a, T>> for ExprsInner<'a> {
            fn from(expr: ExprInner<'a, T>) -> Self {
                unsafe {
                    match T::dtype() {
                        $(
                            $(#[$meta])? DataType::$ty => ExprsInner::$ty(expr.into_dtype::<$real>()),
                        )*
                        _ => unreachable!("Not supported dtype in ExprsInner::from")
                    }
                }
            }
        }

        impl<'a> From<ExprsInner<'a>> for Exprs<'a> {
            #[allow(unreachable_patterns)]
            fn from(expr: ExprsInner<'a>) -> Self {
                match expr {
                    $(
                        $(#[$meta])? ExprsInner::$ty(e) => Exprs::$ty(e.into()),
                    )*
                }
            }
        }
    }
}

impl_from_trait!(
    I32: i32,
    I64: i64,
    F32: f32,
    F64: f64,
    Usize: usize,
    Bool: bool,
    Str: &'a str,
    String: String,
    DateTime: DateTime,
    OptUsize: OptUsize,
    TimeDelta: TimeDelta,
    Object: PyValue,
    VecUsize: Vec<usize>,
    #[cfg(feature = "option_dtype")]
    OptF32: OptF32,
    #[cfg(feature = "option_dtype")]
    OptF64: OptF64,
    #[cfg(feature = "option_dtype")]
    OptI32: OptI32,
    #[cfg(feature = "option_dtype")]
    OptI64: OptI64
);

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
        match_datatype_arm!(all self, e, ExprsInner, T, { e.into_dtype::<T>() })
    }

    /// Match an arm when we have known the dtype.
    ///
    /// # Safety
    ///
    /// `T::dtype()` must match the arm of `Exprs`
    #[allow(unreachable_patterns)]
    pub(super) unsafe fn get<T: ExprElement>(&self) -> &ExprInner<'a, T> {
        match_datatype_arm!(
            all
            self,
            e,
            ExprsInner,
            T,
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
            all
            self,
            e,
            ExprsInner,
            T,
            { std::mem::transmute(e) }
        )
    }

    #[allow(unreachable_patterns)]
    pub(super) fn step(&self) -> usize {
        match_exprs_inner!(self, e, { e.step() })
    }

    // #[allow(unreachable_patterns)]
    // pub(super) fn set_step(&mut self, step: usize) {
    //     match_exprs_inner!(self, e, { e.set_step(step) })
    // }

    // #[allow(unreachable_patterns)]
    // pub(super) fn set_func<T: ExprElement>(&mut self, func: FuncChainType<'a, T>) {
    //     match_exprs_inner!(self, e, { e.set_func(unsafe{std::mem::transmute(func)}) })
    // }

    // #[allow(unreachable_patterns)]
    // pub(super) fn set_name(&mut self, name: Option<String>) {
    //     match_exprs_inner!(self, e, { e.name = name })
    // }

    // #[allow(unreachable_patterns)]
    // pub(super) fn set_owned(&mut self, owned: Option<bool>) {
    //     match_exprs_inner!(self, e, { e.owned = owned })
    // }

    // #[allow(unreachable_patterns)]
    // /// move all the fields from another ExprInner except ExprBase;
    // pub(super) fn move_from<T: ExprElement>(&mut self, other: ExprInner<'a, T>) {
    //     match_exprs_inner!(self, e, {
    //         e.set_step(other.step());
    //         e.set_func(unsafe{std::mem::transmute(other.func)});
    //         e.name = other.name;
    //         e.ref_expr = other.ref_expr;
    //     })
    // }

    // #[allow(unreachable_patterns)]
    // pub(super) fn set_ref_expr(&mut self, ref_expr: Option<Vec<Arc<Mutex<ExprsInner<'a>>>>>) {
    //     match_exprs_inner!(self, e, { e.ref_expr = ref_expr })
    // }

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
    // pub fn try_get_base_context(&self) -> TpResult<ColumnSelector<'a>> {
    //     match_exprs_inner!(self, e, { e.try_get_base_context() })
    // }

    #[allow(unreachable_patterns)]
    pub fn is_context(&self) -> bool {
        match_exprs_inner!(self, e, { e.is_context() })
    }

    // #[allow(unreachable_patterns)]
    // pub fn cast_by_context(&mut self, context: Option<Context<'a>>) -> TpResult<&mut Self> {
    //     if let Some(context) = context {
    //         let exprs = std::mem::take(self);
    //         *self = match_exprs_inner!(exprs, e, { e.cast_by_context(Some(context))?.into() });
    //     }
    //     Ok(self)
    // }

    // #[allow(unreachable_patterns)]
    // pub fn cast_by_context_into(self, context: Option<Context<'a>>) -> TpResult<ExprsInner<'a>> {
    //     match_exprs_inner!(self, e, {e.cast_by_context(context)})
    // }

    #[allow(unreachable_patterns)]
    pub(super) fn view<T: GetDataType>(&self) -> TpResult<ExprOutView<'_, T>> {
        // we have known the datatype of the enum ,so only one arm will be executed
        match_datatype_arm!(
            all
            self,
            e,
            ExprsInner,
            T,
            { e.try_view().map(|view| unsafe { view.into_dtype::<T>() }) }
        )
    }

    #[allow(unreachable_patterns)]
    pub(super) fn eval_inplace(
        &mut self,
        context: Option<Context<'a>>,
    ) -> TpResult<Option<Context<'a>>> {
        match_exprs_inner!(self, e, { e.eval_inplace(context) })
    }
}
