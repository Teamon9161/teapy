#[macro_use]
pub mod exprs;
mod auto_impl_own;
mod expr_view;
mod impl_cmp;
mod impl_mut;
mod impl_ops;
mod impl_own;
mod impl_view;
#[cfg(feature = "blas")]
mod linalg;

pub use exprs::Exprs;
pub use impl_own::DropNaMethod;
#[cfg(feature = "blas")]
pub use linalg::OlsResult;
use pyo3::{Python, ToPyObject};

use super::{
    ArbArray, Arr1, ArrD, ArrOk, ArrViewD, ArrViewMutD, CollectTrustedToVec, DataType, DateTime,
    DefaultNew, EmptyNew, GetDataType, TimeDelta, TimeUnit,
};
use crate::from_py::PyValue;
pub use expr_view::ExprOutView;
use exprs::ExprsInner;
use num::traits::AsPrimitive;
use rayon::prelude::*;
use std::{
    fmt::Debug,
    marker::PhantomData,
    mem,
    sync::{Arc, Mutex},
};

pub trait ExprElement: GetDataType + Default + Sync + Send + Debug {}

impl ExprElement for f32 {}
impl ExprElement for f64 {}
impl ExprElement for i32 {}
impl ExprElement for i64 {}
impl ExprElement for bool {}
impl ExprElement for usize {} // only for index currently
impl ExprElement for String {}
impl<'a> ExprElement for &'a str {}
impl ExprElement for DateTime {}
impl ExprElement for TimeDelta {}

impl ExprElement for Option<usize> {}

#[derive(Default, Clone, Debug)]
pub struct Expr<'a, T: ExprElement> {
    e: Arc<Mutex<ExprsInner<'a>>>,
    _type: PhantomData<T>,
}

impl<'a, T: ExprElement> From<ExprInner<'a, T>> for Expr<'a, T> {
    fn from(e: ExprInner<'a, T>) -> Self {
        Expr::<'a, T> {
            e: Arc::new(Mutex::new(e.into())),
            _type: PhantomData,
        }
    }
}

impl<'a, T: ExprElement + 'a> Expr<'a, T> {
    #[inline]
    pub fn new(arr: ArbArray<'a, T>, name: Option<String>) -> Self {
        ExprInner::<T>::new_with_arr(arr, name).into()
    }

    #[inline]
    pub fn new_from_owned(arr: ArrD<T>, name: Option<String>) -> Self {
        let a: ArbArray<'a, T> = arr.into();
        Self::new(a, name)
    }

    fn downcast(self) -> ExprInner<'a, T> {
        match Arc::try_unwrap(self.e) {
            Ok(e) => {
                let e = e.into_inner().expect("Poison exprs");
                unsafe { e.downcast::<T>() }
            }
            Err(e) => ExprInner::new_with_expr(e, None),
        }
    }

    pub fn name(&self) -> Option<String> {
        unsafe { self.e.lock().unwrap().get::<T>().name() }
    }

    /// Change the name of the Expression immediately
    pub fn rename(&mut self, name: String) {
        unsafe { self.e.lock().unwrap().get_mut::<T>().rename(name) }
    }

    pub fn get_base_type(&self) -> &'static str {
        unsafe { self.e.lock().unwrap().get::<T>().get_base_type() }
    }

    pub fn get_base_strong_count(&self) -> Result<usize, &'static str> {
        unsafe {
            self.e
                .lock()
                .unwrap()
                .get::<T>()
                .get_base_expr_strong_count()
        }
    }

    #[inline(always)]
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.e)
    }

    #[inline(always)]
    pub fn weak_count(&self) -> usize {
        Arc::weak_count(&self.e)
    }

    #[inline(always)]
    pub fn ref_count(&self) -> usize {
        self.strong_count() + self.weak_count()
    }

    #[inline]
    pub fn step(&self) -> usize {
        unsafe { self.e.lock().unwrap().get::<T>().step }
    }

    /// The step of the expression plus the step_acc of the base expression
    pub fn step_acc(&self) -> usize {
        unsafe { self.e.lock().unwrap().get::<T>().step_acc() }
    }

    #[inline]
    pub fn owned(&self) -> Option<bool> {
        unsafe { self.e.lock().unwrap().get::<T>().get_owned() }
    }

    pub fn split_vec_base(self, len: usize) -> Vec<Self>
    where
        T: Clone,
    {
        // todo: improve performance
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            out.push(
                self.clone()
                    .chain_f(move |base| base.into_arr_vec().remove(i).into()),
            );
        }
        out
    }

    /// chain a new function to current function chain, the function accept
    /// an array of type `ExprOut<'_, T>` and return `ExprOut<'a, T2>`
    #[inline]
    pub fn chain_f<F, T2>(self, f: F) -> Expr<'a, T2>
    where
        F: FnOnce(ExprOut<'a, T>) -> ExprOut<'a, T2> + Send + Sync + 'a,
        T2: ExprElement + 'a,
    {
        self.downcast().chain_f(f).into()
    }

    /// chain a new function to current function chain, the function accept
    /// an array of type `ArbArray<'_, T>` and return `ArbArray<'a, T2>`
    #[inline]
    pub fn chain_arr_f<F, T2>(self, f: F) -> Expr<'a, T2>
    where
        F: FnOnce(ArbArray<'a, T>) -> ArbArray<'a, T2> + Send + Sync + 'a,
        T2: ExprElement + 'a,
    {
        self.downcast().chain_arr_f(f).into()
    }

    /// chain a new function to current function chain, the function accept
    /// an array of type `ArrViewD<'a, T>` and return `ArbArray<'a, T>`
    #[inline]
    pub fn chain_view_f<F, T2>(self, f: F) -> Expr<'a, T2>
    where
        F: FnOnce(ArrViewD<'_, T>) -> ArbArray<'a, T2> + Send + Sync + 'a,
        T2: ExprElement + 'a,
    {
        self.downcast().chain_view_f(f).into()
    }

    /// chain a new function to current function chain, the function accept
    /// an array of type `ArbArray<'a, T>` and return `ArbArray<'a, T2>`
    #[inline]
    pub fn chain_view_out_arr_f<F, T2>(self, f: F) -> Expr<'a, T2>
    where
        F: FnOnce(ArbArray<'a, T>) -> ArbArray<'a, T2> + Send + Sync + 'a,
        T2: ExprElement + 'a,
    {
        self.downcast().chain_view_out_arr_f(f).into()
    }

    /// chain a new function to current function chain, the function accept
    /// an array of type ArrViewD<'a, T> and return ArrViewD<'a, T2>
    #[inline]
    pub fn chain_view_out_f<F, T2>(self, f: F) -> Expr<'a, T2>
    where
        F: FnOnce(ArrViewD<'_, T>) -> ArrViewD<'a, T2> + Send + Sync + 'a,
        T2: ExprElement + 'a,
    {
        self.downcast().chain_view_out_f(f).into()
    }

    /// chain a new function to current function chain, the function accept
    /// an array of type `ArrViewMutD<'a, T>` and modify inplace
    #[inline]
    pub fn chain_view_mut_f<F>(self, f: F) -> Expr<'a, T>
    where
        T: Clone,
        F: FnOnce(&mut ArrViewMutD<'_, T>) + Send + Sync + 'a,
    {
        self.downcast().chain_view_mut_f(f).into()
    }

    /// chain a new function to current function chain, the function accept
    /// an array of type `ArrD<'a, T>` and return `ArbArray<'a, T>`
    #[inline]
    pub fn chain_owned_f<F, T2>(self, f: F) -> Expr<'a, T2>
    where
        T: Clone,
        F: FnOnce(ArrD<T>) -> ArbArray<'a, T2> + Send + Sync + 'a,
        T2: ExprElement + 'a,
    {
        self.downcast().chain_owned_f(f).into()
    }

    #[inline]
    pub fn eval(mut self) -> Expr<'a, T> {
        self.eval_inplace();
        self
        // self.downcast().eval().into()
    }

    #[inline]
    pub fn eval_inplace(&mut self) {
        unsafe { self.e.lock().unwrap().get_mut::<T>().eval_inplace() }
    }

    /// create a view of the output array
    #[inline]
    pub fn try_view_arr(&self) -> Result<ArrViewD<'_, T>, &'static str> {
        self.try_view()?.try_into_arr()
    }

    /// create a view of the output array
    #[inline]
    pub fn view_arr(&self) -> ArrViewD<'_, T> {
        self.try_view_arr().expect("can not view as arr")
    }

    #[inline]
    pub fn view(&self) -> ExprOutView<'_, T> {
        self.try_view().unwrap()
    }

    #[inline]
    pub fn try_view(&self) -> Result<ExprOutView<'_, T>, &'static str> {
        let e = self.e.lock().unwrap();
        let view = unsafe { e.get::<T>().try_view()? };
        Ok(unsafe { mem::transmute(view) })
    }

    /// execute the expression and copy the output
    #[inline]
    pub fn value(&mut self) -> ArrD<T>
    where
        T: Clone,
    {
        unsafe { self.e.lock().unwrap().get_mut::<T>().value() }
    }

    #[inline]
    pub fn into_arr(self) -> ArbArray<'a, T> {
        self.downcast().into_arr()
    }

    #[inline]
    pub fn into_out(self) -> ExprOut<'a, T> {
        self.downcast().into_out()
    }

    /// Reinterpret the expression.
    ///
    /// # Safety
    ///
    /// T2 and T must be the same type
    #[inline]
    pub unsafe fn into_dtype<T2: ExprElement>(self) -> Expr<'a, T2> {
        mem::transmute(self)
    }

    #[inline]
    pub fn cast<T2>(self) -> Expr<'a, T2>
    where
        T: AsPrimitive<T2> + Copy + 'static,
        T2: ExprElement + Copy + 'static,
    {
        self.downcast().cast::<T2>().into()
    }

    /// Try casting to bool type
    #[inline]
    pub fn cast_bool(self) -> Expr<'a, bool>
    where
        T: AsPrimitive<i32> + Clone,
    {
        self.downcast().cast_bool().into()
    }

    /// Try casting to string type
    #[inline]
    pub fn cast_string(self) -> Expr<'a, String>
    where
        T: ToString,
    {
        self.downcast().cast_string().into()
    }

    /// Try casting to string type
    #[inline]
    pub fn cast_datetime(self, unit: Option<TimeUnit>) -> Expr<'a, DateTime>
    where
        T: AsPrimitive<i64>,
    {
        self.downcast().cast_datetime(unit).into()
    }

    /// Try casting to string type
    #[inline]
    pub fn cast_timedelta(self) -> Expr<'a, TimeDelta> {
        self.downcast().cast_timedelta().into()
    }

    /// Try casting to object type
    #[inline]
    pub fn cast_object_eager<'py>(self, py: Python<'py>) -> Expr<'a, PyValue>
    where
        T: Clone + ToPyObject,
    {
        self.downcast().cast_object_eager(py).into()
    }
}

pub enum ExprOut<'a, T: ExprElement> {
    Arr(ArbArray<'a, T>),
    ArrVec(Vec<ArbArray<'a, T>>),
    ExprVec(Vec<Expr<'a, T>>),
    #[cfg(feature = "blas")]
    OlsRes(Arc<OlsResult<'a>>),
}

impl<'a, T: ExprElement> ExprOut<'a, T> {
    pub fn into_arr(self) -> ArbArray<'a, T> {
        if let ExprOut::Arr(arr) = self {
            arr
        } else {
            panic!("The output of the expression is not an array")
        }
    }

    #[cfg(feature = "blas")]
    pub fn into_ols_result(self) -> Arc<OlsResult<'a>> {
        if let ExprOut::OlsRes(res) = self {
            res
        } else {
            panic!("The output of the expression is not linear regression result")
        }
    }

    pub fn into_arr_vec(self) -> Vec<ArbArray<'a, T>> {
        match self {
            ExprOut::ArrVec(arr) => arr,
            ExprOut::Arr(arr) => vec![arr],
            _ => panic!("The output of the expression is not a vector of array"),
        }
    }

    pub fn is_owned(&self) -> bool {
        match self {
            ExprOut::Arr(arr) => matches!(arr, ArbArray::Owned(_)),
            #[cfg(feature = "blas")]
            ExprOut::OlsRes(_) => true,
            ExprOut::ArrVec(arr_vec) => {
                let mut out = true;
                for arr in arr_vec {
                    if !matches!(arr, ArbArray::Owned(_)) {
                        out = false
                    }
                }
                out
            }
            ExprOut::ExprVec(_) => false,
        }
    }
}

pub(self) type FuncChainType<'a, T> =
    Box<dyn FnOnce(ExprBase<'a>) -> ExprOut<'a, T> + Send + Sync + 'a>;

impl<'a, T: ExprElement> From<ArbArray<'a, T>> for ExprOut<'a, T> {
    fn from(arr: ArbArray<'a, T>) -> Self {
        ExprOut::Arr(arr)
    }
}

#[cfg(feature = "blas")]
impl<'a, T: ExprElement> From<Arc<OlsResult<'a>>> for ExprOut<'a, T> {
    fn from(res: Arc<OlsResult<'a>>) -> Self {
        ExprOut::OlsRes(res)
    }
}

#[cfg(feature = "blas")]
impl<'a, T: ExprElement> From<OlsResult<'a>> for ExprOut<'a, T> {
    fn from(res: OlsResult<'a>) -> Self {
        ExprOut::OlsRes(Arc::new(res))
    }
}

impl<'a, T: ExprElement> From<Vec<ArbArray<'a, T>>> for ExprOut<'a, T> {
    fn from(res: Vec<ArbArray<'a, T>>) -> Self {
        ExprOut::ArrVec(res)
    }
}

impl<'a, T: ExprElement> From<Vec<ArrViewD<'a, T>>> for ExprOut<'a, T> {
    fn from(res: Vec<ArrViewD<'a, T>>) -> Self {
        let res = res
            .into_iter()
            .map(|arr| ArbArray::View(arr))
            .collect_trusted();
        ExprOut::ArrVec(res)
    }
}

impl<'a, T: ExprElement> From<ArrViewD<'a, T>> for ExprOut<'a, T> {
    fn from(arr: ArrViewD<'a, T>) -> Self {
        ArbArray::View(arr).into()
    }
}

impl<'a, T: ExprElement> From<ArrViewMutD<'a, T>> for ExprOut<'a, T> {
    fn from(arr: ArrViewMutD<'a, T>) -> Self {
        ArbArray::ViewMut(arr).into()
    }
}

impl<'a, T: ExprElement> From<ArrD<T>> for ExprOut<'a, T> {
    fn from(arr: ArrD<T>) -> Self {
        ArbArray::Owned(arr).into()
    }
}

impl<'a, T> DefaultNew for FuncChainType<'a, T>
where
    T: ExprElement,
{
    #[allow(unreachable_patterns)]
    fn default_new() -> Self {
        // Safety: T and the type in each arm are just the same type.
        unsafe {
            match T::dtype() {
                DataType::Bool => Box::new(move |_| {
                    let arr: ArrD<bool> = Default::default();
                    arr.into_dtype::<T>().into()
                }),
                DataType::F64 => Box::new(move |_| {
                    let arr: ArrD<f64> = Default::default();
                    arr.into_dtype::<T>().into()
                }),
                DataType::F32 => Box::new(move |_| {
                    let arr: ArrD<f32> = Default::default();
                    arr.into_dtype::<T>().into()
                }),
                DataType::I32 => Box::new(move |_| {
                    let arr: ArrD<i32> = Default::default();
                    arr.into_dtype::<T>().into()
                }),
                DataType::I64 => Box::new(move |_| {
                    let arr: ArrD<i64> = Default::default();
                    arr.into_dtype::<T>().into()
                }),
                DataType::Usize => Box::new(move |_| {
                    let arr: ArrD<usize> = Default::default();
                    arr.into_dtype::<T>().into()
                }),
                DataType::String => Box::new(move |_| {
                    let arr: ArrD<String> = Default::default();
                    arr.into_dtype::<T>().into()
                }),
                DataType::Str => Box::new(move |_| {
                    let arr: ArrD<&str> = Default::default();
                    arr.into_dtype::<T>().into()
                }),
                DataType::Object => Box::new(move |_| {
                    let arr: ArrD<PyValue> = Default::default();
                    arr.into_dtype::<T>().into()
                }),
                DataType::DateTime => Box::new(move |_| {
                    let arr: ArrD<DateTime> = Default::default();
                    arr.into_dtype::<T>().into()
                }),
                DataType::TimeDelta => Box::new(move |_| {
                    let arr: ArrD<TimeDelta> = Default::default();
                    arr.into_dtype::<T>().into()
                }),
                DataType::OpUsize => Box::new(move |_| {
                    let arr: ArrD<Option<usize>> = Default::default();
                    arr.into_dtype::<T>().into()
                }),
                // _ => unimplemented!("Default Func chain of this type is not implemented")
            }
        }
    }
}

macro_rules! impl_funcchain_new {
    ($expr: expr, $($arm: ident $(($arg: ident))?),*) => {
        unsafe {
            // Safety: T and the type in each arm are just the same type.
            match $expr {
                $(DataType::$arm $(($arg))? => {
                    Box::new(|base| {
                        match base {
                            ExprBase::Arr(arr_dyn) => {
                                if let ArrOk::$arm(arr) = arr_dyn {
                                    arr.into_dtype::<T>().into()
                                } else {
                                    unreachable!()
                                }
                            },
                            ExprBase::ArrVec(arr_vec) => {
                                let out = arr_vec.into_iter().map(|arr| {
                                    if let ArrOk::$arm(arr) = arr {
                                        arr.into_dtype::<T>()
                                    } else {
                                        panic!("Create expression base with a vector of array only support one dtype.")
                                    }
                                }).collect_trusted();
                                out.into()
                            },
                            ExprBase::ArcArr(arc_arr) => {
                                // we should not modify the base expression,
                                // so just create a view of the array
                                let out: ExprOut<'_, T> = arc_arr.view::<T>().into();
                                return mem::transmute(out)
                            },
                            ExprBase::Expr(exprs) => {
                                let ref_count = Arc::strong_count(&exprs);
                                let mut exprs_lock = exprs.lock().unwrap();
                                // if exprs_lock.step() != 0 {
                                exprs_lock.eval_inplace();
                                // }
                                // we should not modify the base expression
                                // safety: the data of the base expression must exist
                                if ref_count > 1 {
                                    // panic!("ref_count = {} > 1", ref_count);
                                    let out: ExprOut<'_, T> = exprs_lock.view::<T>().unwrap().into();
                                    return mem::transmute(out)
                                } else {
                                    mem::drop(exprs_lock);
                                    Expr {e:exprs, _type: PhantomData}.into_out()
                                }
                            },
                            ExprBase::ExprVec(mut expr_vec) => {
                                expr_vec.iter_mut().for_each(|e| {
                                    let mut expr_lock = e.lock().unwrap();
                                    // if expr_lock.step() != 0 {
                                    expr_lock.eval_inplace();
                                    // }
                                });
                                let expr_vec = expr_vec.into_iter().map(|e| {
                                    Expr {e, _type: PhantomData}
                                }).collect_trusted();
                                ExprOut::ExprVec(expr_vec)
                            },
                            #[cfg(feature="blas")] ExprBase::OlsRes(res) => res.into()
                        }
                    })
                },)*
                _ => unimplemented!("Not support dtype of function chain")
            }
        }
    };
}

impl<'a, T> EmptyNew for FuncChainType<'a, T>
where
    T: ExprElement,
{
    #[allow(unreachable_patterns)]
    fn empty_new() -> Self {
        impl_funcchain_new!(
            T::dtype(),
            Bool,
            F32,
            F64,
            I64,
            I32,
            Usize,
            String,
            Str,
            Object,
            DateTime,
            TimeDelta,
            OpUsize
        )
    }
}

pub(self) struct ExprInner<'a, T: ExprElement> {
    base: ExprBase<'a>,
    func: FuncChainType<'a, T>,
    pub step: usize,
    pub owned: Option<bool>,
    pub name: Option<String>,
    ref_expr: Option<Vec<Arc<Mutex<ExprsInner<'a>>>>>,
}

impl<'a, T: ExprElement> Debug for ExprInner<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Expr")
            .field("name", &self.name)
            .field("step", &self.step)
            .field("owned", &self.owned)
            .field("base", &self.base)
            .finish()
    }
}

impl<'a, T> Default for ExprInner<'a, T>
where
    T: ExprElement,
{
    fn default() -> Self {
        ExprInner {
            base: Default::default(),
            func: DefaultNew::default_new(),
            step: 0,
            owned: None,
            name: None,
            ref_expr: None,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub(self) enum ExprBase<'a> {
    Expr(Arc<Mutex<ExprsInner<'a>>>), // an expression based on another expression
    ExprVec(Vec<Arc<Mutex<ExprsInner<'a>>>>), // an expression based on expressions
    Arr(ArrOk<'a>),                   // classical expression based on an array
    ArrVec(Vec<ArrOk<'a>>),           // an expression based on a vector of array
    ArcArr(Arc<ArrOk<'a>>),           // multi expressions share the same array
    #[cfg(feature = "blas")]
    OlsRes(Arc<OlsResult<'a>>), // only for least squares
}

impl Default for ExprBase<'_> {
    fn default() -> Self {
        ExprBase::Arr(Default::default())
    }
}

impl<'a, T: ExprElement> ExprInner<'a, T> {
    // #[allow(dead_code)]
    // pub fn new(arr: ExprOut<'a, T>, name: Option<String>) -> Self {
    //     match arr {
    //         ExprOut::Arr(arr) => Self::new_with_arr(arr, name),
    //         #[cfg(feature = "blas")]
    //         ExprOut::OlsRes(res) => ExprInner::<T> {
    //             base: ExprBase::OlsRes(res),
    //             func: EmptyNew::empty_new(),
    //             step: 0,
    //             owned: None,
    //             name,
    //             ref_expr: None,
    //         },
    //         ExprOut::ArrVec(_) => {
    //             unimplemented!("Create an expression with a vector of array is not implemented")
    //         }
    //         ExprOut::ExprVec(expr_vec) => {
    //             let expr_vec = expr_vec.into_iter().map(|e| e.e).collect_trusted();
    //             Self::new_with_expr_vec(expr_vec, name)
    //         }
    //     }
    // }

    // pub fn new_with_base(arr: ExprBase<'a>, name: Option<String>) -> Self {
    //     ExprInner::<T> {
    //         base: arr,
    //         func: EmptyNew::empty_new(),
    //         step: 0,
    //         owned: None,
    //         name,
    //         ref_expr: None,
    //     }
    // }

    pub fn new_with_arr(arr: ArbArray<'a, T>, name: Option<String>) -> Self {
        ExprInner::<T> {
            base: ExprBase::Arr(arr.into()),
            func: EmptyNew::empty_new(),
            step: 0,
            owned: None,
            name,
            ref_expr: None,
        }
    }

    #[allow(dead_code)]
    pub fn new_with_arc_arr(arr: Arc<ArrOk<'a>>, name: Option<String>) -> Self {
        ExprInner::<T> {
            base: ExprBase::ArcArr(arr),
            func: EmptyNew::empty_new(),
            step: 0,
            owned: None,
            name,
            ref_expr: None,
        }
    }

    // Create a new expression which is based on a current expression
    pub fn new_with_expr(expr: Arc<Mutex<ExprsInner<'a>>>, name: Option<String>) -> Self {
        // let step: usize;
        // let owned: Option<bool>;
        let name = {
            let e = expr.lock().unwrap();
            let e_ = unsafe { e.get::<T>() };
            // (step, owned) = (e_.step, e_.owned);
            // owned = e_.owned;
            if name.is_none() {
                e_.name()
            } else {
                name
            }
        };

        ExprInner::<T> {
            base: ExprBase::Expr(expr),
            func: EmptyNew::empty_new(),
            owned: Some(false),
            step: 0,
            // step,
            name,
            ref_expr: None,
        }
    }

    #[allow(dead_code)]
    pub fn new_with_expr_vec(
        expr_vec: Vec<Arc<Mutex<ExprsInner<'a>>>>,
        name: Option<String>,
    ) -> Self {
        ExprInner::<T> {
            base: ExprBase::ExprVec(expr_vec),
            func: EmptyNew::empty_new(),
            owned: Some(false),
            step: 0,
            name,
            ref_expr: None,
        }
    }
    /// Reinterpret the expression.
    ///
    /// # Safety
    ///
    /// T2 and T must be the same type
    pub unsafe fn into_dtype<T2: ExprElement>(self) -> ExprInner<'a, T2> {
        mem::transmute(self)
    }

    #[inline(always)]
    pub fn step(&self) -> usize {
        self.step
    }

    /// The step of the expression plus the step_acc of the base expression
    pub fn step_acc(&self) -> usize {
        let base_acc_step = match &self.base {
            ExprBase::Expr(eb) => eb.lock().unwrap().step_acc(),
            ExprBase::ExprVec(ebs) => {
                let mut acc = 0;
                ebs.iter().for_each(|e| acc += e.lock().unwrap().step_acc());
                acc
            }
            _ => 0,
        };
        self.step + base_acc_step
    }

    /// get name of the expression
    #[inline(always)]
    pub fn name(&self) -> Option<String> {
        self.name.clone()
    }

    /// rename inplace
    #[inline(always)]
    pub fn rename(&mut self, name: String) {
        self.name = Some(name)
    }

    pub fn get_base_type(&self) -> &'static str {
        match &self.base {
            ExprBase::Expr(_) => "Expr",
            ExprBase::ExprVec(_) => "ExprVec",
            ExprBase::Arr(arr) => arr.get_type(),
            ExprBase::ArrVec(_) => "ArrVec",
            ExprBase::ArcArr(_) => "ArcArr",
            #[cfg(feature = "blas")]
            ExprBase::OlsRes(_) => "OlsRes",
        }
    }

    pub fn get_base_expr_strong_count(&self) -> Result<usize, &'static str> {
        if let ExprBase::Expr(expr) = &self.base {
            Ok(Arc::strong_count(expr))
        } else {
            Err("The base of the expression is not Expr")
        }
    }

    #[inline(always)]
    pub fn set_base(&mut self, base: ExprBase<'a>) {
        self.base = base;
    }

    #[inline(always)]
    pub fn set_base_by_arr(&mut self, base: ArrOk<'a>) {
        self.set_base(ExprBase::Arr(base));
    }

    #[inline(always)]
    pub fn set_base_by_arr_vec(&mut self, base: Vec<ArrOk<'a>>) {
        self.set_base(ExprBase::ArrVec(base));
    }

    #[cfg(feature = "blas")]
    #[inline(always)]
    pub fn set_base_by_ols_res(&mut self, base: Arc<OlsResult<'a>>) {
        self.set_base(ExprBase::OlsRes(base));
    }

    #[inline(always)]
    pub fn set_step(&mut self, step: usize) {
        self.step = step;
    }

    #[inline(always)]
    pub fn set_func(&mut self, func: FuncChainType<'a, T>) {
        self.func = func;
    }

    #[inline(always)]
    pub fn get_owned(&self) -> Option<bool> {
        self.owned
    }

    #[inline(always)]
    pub fn set_owned(&mut self, owned: bool) {
        self.owned = Some(owned);
    }

    #[inline(always)]
    /// Reset the function chain.
    pub fn reset_func(&mut self) {
        self.set_func(EmptyNew::empty_new());
    }

    #[inline(always)]
    /// Reset the function chain and the step.
    pub fn reset(&mut self) {
        self.set_step(0);
        self.reset_func();
    }

    /// Execute the expression and get a new Expr with step 0
    #[inline]
    pub fn eval(mut self) -> Self {
        self.eval_inplace();
        self
    }

    /// Execute the expression inplace
    #[inline]
    pub fn eval_inplace(&mut self) {
        if self.step() != 0 {
            let default_func: FuncChainType<'a, T> = DefaultNew::default_new();
            let func = mem::replace(&mut self.func, default_func);
            if let ExprBase::Expr(base_expr) = &self.base {
                if let Some(is_owned) = self.get_owned() {
                    if !is_owned {
                        self.ref_expr = Some(vec![base_expr.clone()])
                    }
                }
            }
            let base = mem::take(&mut self.base);
            let out = func(base);
            self.set_owned(out.is_owned());
            if self.get_owned().unwrap() {
                self.ref_expr = None;
            }
            match out {
                ExprOut::Arr(arr) => self.set_base_by_arr(arr.into()),
                ExprOut::ArrVec(arr_vec) => self.set_base_by_arr_vec(
                    arr_vec.into_iter().map(|arr| arr.into()).collect_trusted(),
                ),
                ExprOut::ExprVec(_expr_vec) => unimplemented!(
                    "Create a new expression with an vector of expression is not supported yet."
                ),
                #[cfg(feature = "blas")]
                ExprOut::OlsRes(res) => self.set_base_by_ols_res(res),
            }
        } else {
            // step is zero but we should check the step of the expression base
            match &self.base {
                ExprBase::Expr(eb) => {
                    if let Some(is_owned) = self.get_owned() {
                        if !is_owned {
                            self.ref_expr = Some(vec![eb.clone()])
                        }
                    }
                    eb.lock().unwrap().eval_inplace()
                }
                // we assume that the result are not based on the expressions
                ExprBase::ExprVec(ebs) => ebs
                    .into_par_iter()
                    .for_each(|eb| eb.lock().unwrap().eval_inplace()),
                _ => {}
            }
        }
        self.reset();
    }

    /// Get a view of the output, raise error when the expression wasn't executed.
    pub fn try_view(&self) -> Result<ExprOutView<'_, T>, &'static str> {
        if self.step > 0 {
            return Err("Expression has not been executed.");
        }
        match &self.base {
            ExprBase::Arr(arr_base) => unsafe { Ok(arr_base.view::<T>().into()) },
            ExprBase::ArcArr(arc_arr) => unsafe { Ok(arc_arr.view::<T>().into()) },
            ExprBase::ArrVec(arr_vec) => unsafe {
                Ok(arr_vec
                    .iter()
                    .map(|arr| arr.view::<T>())
                    .collect_trusted()
                    .into())
            },
            ExprBase::Expr(expr) => {
                let expr_lock = expr.lock().unwrap();
                if expr_lock.step() == 0 {
                    unsafe { Ok(mem::transmute(expr_lock.view::<T>()?)) }
                } else {
                    Err("Base Expression has not been executed.")
                }
            }
            ExprBase::ExprVec(_) => {
                unimplemented!("view a vector of expression is not supported yet.")
            }
            #[cfg(feature = "blas")]
            ExprBase::OlsRes(res) => {
                let res: ExprOutView<'a, T> = res.clone().into();
                Ok(unsafe { mem::transmute(res) })
            }
        }
    }

    // /// Execute the expression and get a view of the output
    // pub fn view(&self) -> ExprOutView<'_, T> {
    //     self.try_view().expect("Expression has not been executed.")
    // }

    /// Execute the expression and get a view of array output
    ///
    /// # Safety
    ///
    /// The data of the array view must exists.
    #[inline]
    pub fn try_view_arr(&self) -> Result<ArrViewD<'_, T>, &'static str> {
        self.try_view()?.try_into_arr()
    }

    /// Execute the expression and get a view of array output
    ///
    /// # Safety
    ///
    /// The data of the array view must exists.
    #[inline]
    #[allow(dead_code)]
    pub fn try_view_arr_vec(&self) -> Result<Vec<ArrViewD<'_, T>>, &'static str> {
        self.try_view()?.try_into_arr_vec()
    }
    /// Execute the expression and get a view of array output
    ///
    /// # Safety
    ///
    /// The data of the array view must exists.
    #[inline]
    pub fn view_arr(&self) -> ArrViewD<'_, T> {
        self.try_view_arr().expect("Can not view as an array")
    }

    /// execute the expression and copy the output
    #[inline]
    pub fn value(&mut self) -> ArrD<T>
    where
        T: Clone,
    {
        self.eval_inplace();
        self.view_arr().to_owned()
    }

    #[inline]
    pub fn into_arr(self) -> ArbArray<'a, T> {
        self.into_out().into_arr()
    }

    #[inline]
    pub fn into_out(self) -> ExprOut<'a, T> {
        (self.func)(self.base)
    }
    /// chain a new function to current function chain, the function accept
    /// an array of type `ExprOut<'a, T>` and return `ExprOut<'a, T>`
    pub fn chain_f<F, T2>(self, f: F) -> ExprInner<'a, T2>
    where
        F: FnOnce(ExprOut<'a, T>) -> ExprOut<'a, T2> + Send + Sync + 'a,
        T2: ExprElement,
    {
        ExprInner::<'a, T2> {
            base: self.base,
            step: self.step + 1,
            name: self.name,
            owned: self.owned, // We don't know if the expression has ownership of the data after adding a function
            func: Box::new(move |base: ExprBase<'a>| f((self.func)(base))),
            ref_expr: None,
        }
    }

    /// chain a new function to current function chain, the function accept
    /// an array of type `ArbArray<'a, T>` and return `ArbArray<'a, T>`
    pub fn chain_arr_f<F, T2>(self, f: F) -> ExprInner<'a, T2>
    where
        F: FnOnce(ArbArray<'a, T>) -> ArbArray<'a, T2> + Send + Sync + 'a,
        T2: ExprElement,
    {
        ExprInner::<'a, T2> {
            base: self.base,
            step: self.step + 1,
            name: self.name,
            owned: self.owned,
            func: Box::new(move |base: ExprBase<'a>| f((self.func)(base).into_arr()).into()),
            ref_expr: None,
        }
    }

    /// chain a new function to current function chain, the function accept
    /// an array of type `ArrViewD<'a, T>` and return `ArbArray<'a, T>`
    pub fn chain_view_f<F, T2>(self, f: F) -> ExprInner<'a, T2>
    where
        F: FnOnce(ArrViewD<'_, T>) -> ArbArray<'a, T2> + Send + Sync + 'a,
        T2: ExprElement,
    {
        ExprInner::<'a, T2> {
            base: self.base,
            step: self.step + 1,
            name: self.name,
            owned: self.owned,
            func: Box::new(move |base: ExprBase<'a>| {
                let arb_array = (self.func)(base).into_arr();
                match arb_array {
                    ArbArray::View(arr) => f(arr).into(),
                    ArbArray::ViewMut(arr) => f(arr.view()).into(),
                    ArbArray::Owned(arr) => f(arr.view()).into(),
                }
            }),
            ref_expr: None,
        }
    }

    /// chain a new function to current function chain, the function accept
    /// an array of type `ArbArray<'a, T>` and return `ArbArray<'a, T2>`
    pub fn chain_view_out_arr_f<F, T2>(mut self, f: F) -> ExprInner<'a, T2>
    where
        F: FnOnce(ArbArray<'a, T>) -> ArbArray<'a, T2> + Send + Sync + 'a,
        T2: ExprElement,
    {
        ExprInner::<'a, T2> {
            base: self.base,
            step: self.step + 1,
            name: self.name.clone(),
            owned: Some(false),
            func: Box::new(move |base: ExprBase<'a>| {
                // let base_expr: ExprsInner = ExprInner::<T>::new_with_base(base, self.name).into();
                // let base_expr = Arc::new(Mutex::new(base_expr));
                // if let Some(mut ref_expr) = self.ref_expr {
                //     ref_expr.push(base_expr.clone());
                //     self.ref_expr = Some(ref_expr)
                // } else {
                //     self.ref_expr = Some(vec![base_expr.clone()])
                // }
                // unsafe{base_expr.lock().unwrap().get_mut::<T>().eval_inplace()};
                let arb_array = (self.func)(base).into_arr();
                let last_out = if matches!(arb_array, ArbArray::Owned(_)) {
                    let base_expr: ExprsInner =
                        ExprInner::<T>::new_with_arr(arb_array, self.name).into();
                    let base_expr = Arc::new(Mutex::new(base_expr));
                    if let Some(mut ref_expr) = self.ref_expr {
                        ref_expr.push(base_expr);
                        self.ref_expr = Some(ref_expr);
                    } else {
                        self.ref_expr = Some(vec![base_expr]);
                    }
                    if let Some(ref_expr) = &self.ref_expr {
                        // this should always be true
                        let e = ref_expr.last().unwrap().lock().unwrap();
                        let arr_view = e.view::<T>().unwrap().into_arr();
                        let arr_view: ArrViewD<'a, T> = unsafe { mem::transmute(arr_view) };
                        arr_view.into()
                    } else {
                        unreachable!()
                    }
                } else {
                    arb_array
                };
                f(last_out).into()
            }),
            ref_expr: None,
        }
    }

    /// chain a new function to current function chain, the function accept
    /// an array of type `ArrViewD<'a, T>` and return `ArrViewD<'a, T2>`
    pub fn chain_view_out_f<F, T2>(mut self, f: F) -> ExprInner<'a, T2>
    where
        F: FnOnce(ArrViewD<'_, T>) -> ArrViewD<'a, T2> + Send + Sync + 'a,
        T2: ExprElement,
    {
        ExprInner::<'a, T2> {
            base: self.base,
            step: self.step + 1,
            name: self.name.clone(),
            owned: Some(false),
            func: Box::new(move |base: ExprBase<'a>| {
                // let base_expr: ExprsInner = ExprInner::<T>::new_with_base(base, self.name).into();
                // let base_expr = Arc::new(Mutex::new(base_expr));
                // if let Some(mut ref_expr) = self.ref_expr {
                //     ref_expr.push(base_expr.clone());
                //     self.ref_expr = Some(ref_expr)
                // } else {
                //     self.ref_expr = Some(vec![base_expr.clone()])
                // }
                // unsafe{base_expr.lock().unwrap().get_mut::<T>().eval_inplace()};
                let arb_array = (self.func)(base).into_arr();
                // match arb_array {
                //     ArbArray::View(arr) => f(arr).into(),
                //     ArbArray::ViewMut(arr) => f(arr.view()).into(),
                //     ArbArray::Owned(arr) => f(arr.view()).into(),
                // }
                let last_out = if matches!(arb_array, ArbArray::Owned(_)) {
                    let base_expr: ExprsInner =
                        ExprInner::<T>::new_with_arr(arb_array, self.name).into();
                    let base_expr = Arc::new(Mutex::new(base_expr));
                    if let Some(mut ref_expr) = self.ref_expr {
                        ref_expr.push(base_expr);
                        self.ref_expr = Some(ref_expr);
                    } else {
                        self.ref_expr = Some(vec![base_expr]);
                    }
                    if let Some(ref_expr) = &self.ref_expr {
                        // this should always be true
                        let e = ref_expr.last().unwrap().lock().unwrap();
                        let arr_view = e.view::<T>().unwrap().into_arr();
                        let arr_view: ArrViewD<'a, T> = unsafe { mem::transmute(arr_view) };
                        arr_view.into()
                    } else {
                        unreachable!()
                    }
                } else {
                    arb_array
                };
                match last_out {
                    ArbArray::View(arr) => f(arr).into(),
                    ArbArray::ViewMut(arr) => f(arr.view()).into(),
                    ArbArray::Owned(arr) => f(arr.view()).into(),
                }
            }),
            ref_expr: None,
        }
    }

    /// chain a new function to current function chain, the function accept
    /// an array of type `ArrViewMutD<'a, T>` and modify inplace.
    pub fn chain_view_mut_f<F>(self, f: F) -> ExprInner<'a, T>
    where
        T: Clone,
        F: FnOnce(&mut ArrViewMutD<'_, T>) + Send + Sync + 'a,
    {
        ExprInner::<'a, T> {
            base: self.base,
            step: self.step + 1,
            name: self.name,
            owned: self.owned,
            func: Box::new(move |base: ExprBase<'a>| {
                let arb_array = (self.func)(base).into_arr();
                match arb_array {
                    ArbArray::View(arr) => {
                        let mut arr = arr.to_owned();
                        f(&mut arr.view_mut());
                        arr.into()
                    }
                    ArbArray::ViewMut(mut arr) => {
                        f(&mut arr);
                        arr.into()
                    }
                    ArbArray::Owned(mut arr) => {
                        f(&mut arr.view_mut());
                        arr.into()
                    }
                }
            }),
            ref_expr: None,
        }
    }

    /// chain a new function to current function chain, the function accept
    /// an array of type `ArrD<'a, T>` and return `ArbArray<'a, T>`
    pub fn chain_owned_f<F, T2>(self, f: F) -> ExprInner<'a, T2>
    where
        T: Clone,
        F: FnOnce(ArrD<T>) -> ArbArray<'a, T2> + Send + Sync + 'a,
        T2: ExprElement,
    {
        ExprInner::<'a, T2> {
            base: self.base,
            step: self.step + 1,
            name: self.name,
            owned: self.owned,
            func: Box::new(move |base: ExprBase<'a>| {
                let arb_array = (self.func)(base).into_arr();
                match arb_array {
                    ArbArray::View(arr) => f(arr.to_owned()).into(),
                    ArbArray::ViewMut(arr) => f(arr.to_owned()).into(),
                    ArbArray::Owned(arr) => f(arr).into(),
                }
            }),
            ref_expr: None,
        }
    }

    /// Cast to another type
    pub fn cast<T2>(self) -> ExprInner<'a, T2>
    where
        T: AsPrimitive<T2> + Copy + 'static,
        T2: ExprElement + Copy + 'static,
    {
        if T::dtype() == T2::dtype() {
            // safety: T and T2 are the same type
            unsafe { mem::transmute(self) }
        } else {
            self.chain_view_f(move |arr| arr.cast::<T2>().into())
        }
    }

    /// Try casting to bool type
    pub fn cast_bool(self) -> ExprInner<'a, bool>
    where
        T: AsPrimitive<i32>,
    {
        if T::dtype() == DataType::Bool {
            // safety: T and T2 are the same type
            unsafe { mem::transmute(self) }
        } else {
            self.chain_view_f(move |arr| arr.to_bool().into())
        }
    }

    /// Try casting to datetime type
    #[allow(clippy::unnecessary_unwrap)]
    pub fn cast_datetime(self, unit: Option<TimeUnit>) -> ExprInner<'a, DateTime>
    where
        T: AsPrimitive<i64>,
    {
        if (T::dtype() == DataType::DateTime) || unit.is_none() {
            unsafe { mem::transmute(self) }
        } else {
            self.chain_view_f(move |arr| arr.to_datetime(unit.unwrap()).into())
        }
    }

    pub fn cast_timedelta(self) -> ExprInner<'a, TimeDelta> {
        if T::dtype() == DataType::TimeDelta {
            unsafe { mem::transmute(self) }
        } else if T::dtype() == DataType::String {
            unsafe {
                self.into_dtype::<String>()
                    .chain_view_f(move |arr| arr.map(|s| TimeDelta::parse(s.as_str())).into())
            }
        } else {
            unimplemented!("can not cast to timedelta directly")
        }
    }

    /// Try casting to string type
    pub fn cast_string(self) -> ExprInner<'a, String>
    where
        T: ToString,
    {
        if T::dtype() == DataType::String {
            // safety: T and T2 are the same type
            unsafe { mem::transmute(self) }
        } else {
            self.chain_view_f(move |arr| arr.to_string().into())
        }
    }

    /// Try casting to bool type
    pub fn cast_object_eager(self, py: Python) -> ExprInner<'a, PyValue>
    where
        T: ToPyObject + Clone,
    {
        if T::dtype() == DataType::Object {
            // safety: T and T2 are the same type
            unsafe { mem::transmute(self) }
        } else {
            let e = self.eval();
            let name = e.name();
            ExprInner::new_with_arr(e.view_arr().to_object(py).into(), name)
        }
    }
}

impl<'a, T: ExprElement + 'a> From<T> for Expr<'a, T> {
    fn from(v: T) -> Self {
        Expr::new(v.into(), None)
    }
}

impl<'a, T: ExprElement + 'a> From<Vec<T>> for Expr<'a, T> {
    fn from(vec: Vec<T>) -> Self {
        let e: ExprInner<'a, T> = vec.into();
        e.into()
    }
}

impl<'a, T: ExprElement + 'a> From<Vec<T>> for ExprInner<'a, T> {
    fn from(vec: Vec<T>) -> Self {
        let arr = Arr1::from_vec(vec).to_dimd().unwrap();
        ExprInner::new_with_arr(arr.into(), None)
    }
}
