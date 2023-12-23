use super::data::Data;
use super::expr_element::ExprElement;
use super::expr_inner::{ExprInner, FuncOut};
use super::FuncNode;
use crate::Context;
#[cfg(feature = "blas")]
use crate::OlsResult;
use core::datatype::{GetDataType, GetNone};
use core::{
    error::TpResult,
    match_all, match_arrok,
    prelude::{ArbArray, ArrD, ArrOk, ArrViewD},
    utils::CollectTrustedToVec,
};
use parking_lot::Mutex;
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;
// use serde::Serialize;

#[derive(Default)]
pub struct Expr<'a>(Arc<Mutex<ExprInner<'a>>>);

// impl<'a> Serialize for Expr<'a> {
//     fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
//     where
//         S: serde::Serializer,
//         {
//             let e = self.lock();
//             let data = e.view_data(None).unwrap();
//             data.serialize(serializer)
//         }
// }

impl<'a> Clone for Expr<'a> {
    #[inline]
    fn clone<'b>(&'b self) -> Expr<'a> {
        let name = self.name();
        let inner = ExprInner::new(Self(self.0.clone()).into(), name);
        Expr(Arc::new(Mutex::new(inner)))
    }
}

impl<'a, T> From<Option<T>> for Expr<'a>
where
    T: ExprElement + GetNone + 'a,
{
    #[inline]
    fn from(v: Option<T>) -> Self {
        let v = v.unwrap_or_else(T::none);
        v.into()
    }
}

impl<'a, T: ExprElement + 'a> From<T> for Expr<'a> {
    #[inline]
    fn from(arr: T) -> Self {
        let e: ExprInner<'a> = arr.into();
        e.into()
    }
}

impl<'a, T: ExprElement + 'a> From<ArrD<T>> for Expr<'a> {
    #[inline]
    fn from(arr: ArrD<T>) -> Self {
        let e: ExprInner<'a> = arr.into();
        e.into()
    }
}

impl<'a, T: ExprElement + 'a> From<ArbArray<'a, T>> for Expr<'a> {
    #[inline]
    fn from(arr: ArbArray<'a, T>) -> Self {
        let e: ExprInner<'a> = arr.into();
        e.into()
    }
}

impl<'a> From<ExprInner<'a>> for Expr<'a> {
    #[inline]
    fn from(value: ExprInner<'a>) -> Self {
        Expr(Arc::new(Mutex::new(value)))
    }
}

impl<'a> From<Data<'a>> for Expr<'a> {
    #[inline]
    fn from(data: Data<'a>) -> Self {
        Expr(Arc::new(Mutex::new(ExprInner::new(data, None))))
    }
}

impl<'a> From<ArrOk<'a>> for Expr<'a> {
    #[inline]
    fn from(arr: ArrOk<'a>) -> Self {
        let data: Data = arr.into();
        data.into()
    }
}

impl<'a, T: GetDataType + 'a> From<ArrViewD<'a, T>> for Expr<'a> {
    #[inline]
    fn from(arr: ArrViewD<'a, T>) -> Self {
        let data: Data = arr.into();
        data.into()
    }
}

impl<'a> Deref for Expr<'a> {
    type Target = Arc<Mutex<ExprInner<'a>>>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> Debug for Expr<'a> {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.lock())
    }
}

impl<'a> Expr<'a> {
    #[inline]
    pub fn new_from_owned<T: ExprElement + 'a>(arr: ArrD<T>, name: Option<String>) -> Self {
        let e: ExprInner<'a> = arr.into();
        let mut e: Expr<'a> = e.into();
        e.set_name(name);
        e
    }

    #[inline]
    pub fn new_from_arr(arr: ArrOk<'a>, name: Option<String>) -> Self {
        let mut e: Expr<'a> = arr.into();
        e.set_name(name);
        e
    }

    #[inline]
    pub fn new<T: ExprElement + 'a>(arr: ArbArray<'a, T>, name: Option<String>) -> Self {
        let e: ExprInner<'a> = arr.into();
        let mut e: Expr<'a> = e.into();
        e.set_name(name);
        e
    }

    #[inline]
    pub fn into_inner(self) -> Result<ExprInner<'a>, Self> {
        let e = Arc::try_unwrap(self.0).map_err(Self)?;
        Ok(e.into_inner())
    }

    // pub fn clone_to_life<'b>(&self) -> Expr<'b>
    // where:
    //     {}

    #[inline(always)]
    pub fn step_acc(&self) -> usize {
        self.lock().step_acc()
    }

    #[inline(always)]
    pub fn step(&self) -> usize {
        self.lock().step()
    }

    #[inline(always)]
    pub fn is_owned(&self) -> bool {
        self.lock().is_owned()
    }

    #[inline(always)]
    pub fn dtype(&self) -> String {
        self.lock().dtype()
    }

    #[inline(always)]
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }

    #[inline(always)]
    pub fn name(&self) -> Option<String> {
        self.lock().name_owned()
    }

    #[inline]
    pub fn ref_name(&self) -> Option<&str> {
        let l = self.lock();
        let name = l.name();
        unsafe { std::mem::transmute(name) }
    }

    #[inline(always)]
    pub fn set_name(&mut self, name: Option<String>) {
        self.lock().set_name(name);
    }

    #[inline(always)]
    pub fn rename(&mut self, name: String) {
        self.lock().set_name(Some(name))
    }

    #[inline(always)]
    pub fn base_type(&self) -> &'static str {
        self.lock().base_type()
    }

    #[inline]
    pub fn context_clone(&self) -> Self {
        let inner = self.lock().context_clone();
        inner.into()
    }

    #[inline]
    pub fn chain_f_ctx<F>(&mut self, f: F)
    where
        F: Fn(FuncOut<'a>) -> TpResult<FuncOut<'a>> + Send + Sync + 'a,
    {
        if let Some(e) = Arc::get_mut(&mut self.0) {
            e.get_mut().chain_f_ctx(f);
        } else {
            // if we just lock the base and evaluate, there may be a deadlock
            *self = self.clone();
            if let Some(e) = Arc::get_mut(&mut self.0) {
                e.get_mut().chain_f_ctx(f);
            } else {
                unreachable!("Arc::get_mut failed")
            }
        }
    }

    #[inline]
    pub fn simplify(&mut self) {
        if let Some(e) = Arc::get_mut(&mut self.0) {
            e.get_mut().simplify()
        } else {
            self.0.lock().simplify()
        }
    }

    #[inline]
    pub fn eval_inplace(&mut self, ctx: Option<Context<'a>>) -> TpResult<&mut Self> {
        if let Some(e) = Arc::get_mut(&mut self.0) {
            e.get_mut().eval_inplace(ctx, false)?;
        } else {
            self.0.lock().eval_inplace(ctx, false)?;
        }
        Ok(self)
    }

    /// this will freeze the context of the expression,
    /// the expression can not be evaluated again in other context
    #[inline]
    pub fn eval_inplace_freeze(&mut self, ctx: Option<Context<'a>>) -> TpResult<&mut Self> {
        if let Some(e) = Arc::get_mut(&mut self.0) {
            e.get_mut().eval_inplace(ctx, true)?;
        } else {
            self.0.lock().eval_inplace(ctx, true)?;
        }
        Ok(self)
    }

    // pub fn cxt_ref_to_owned(&mut self) -> TpResult<&mut Self> {
    //     if let Some(e) = Arc::get_mut(&mut self.0) {
    //         e.get_mut().ctx_ref_to_owned()?;
    //     } else {
    //         // unreachable!("the result may be incorrect in context evaluation")
    //         self.0.lock().ctx_ref_to_owned()?;
    //     }
    //     Ok(self)
    // }
    // pub fn into_out(self, ctx: Option<Context<'a>>) -> TpResult<Data<'a>> {
    //     match Arc::try_unwrap(self.0) {
    //         Ok(inner) => inner.into_inner().into_out(ctx),
    //         Err(expr) => {
    //             let mut e = expr.lock();
    //             e.eval_inplace(ctx)?;
    //             let data = e.view_data()?;
    //             Ok(data.clone())
    //             // let arr_view = arr.view();
    //             // // safety: the expression has been evaluated
    //             // Ok(unsafe{std::mem::transmute(arr_view)})
    //         }
    //     }
    // }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn into_arr(self, ctx: Option<Context<'a>>) -> TpResult<ArrOk<'a>> {
        match Arc::try_unwrap(self.0) {
            Ok(inner) => inner.into_inner().into_arr(ctx),
            Err(expr) => {
                let out = {
                    let mut e = expr.lock();
                    e.eval_inplace(ctx.clone(), false)?;
                    let arr = e.view_arr(ctx.as_ref())?;
                    // need clone here
                    match_arrok!(arr, a, { a.view().to_owned().into() })
                };
                Ok(out)
            }
        }
    }

    #[allow(unreachable_patterns)]
    #[cfg(feature = "blas")]
    #[inline]
    pub fn into_ols_res(self, ctx: Option<Context<'a>>) -> TpResult<Arc<OlsResult<'a>>> {
        match Arc::try_unwrap(self.0) {
            Ok(inner) => inner.into_inner().into_ols_res(ctx),
            Err(expr) => {
                let mut e = expr.lock();
                e.eval_inplace(ctx.clone(), false)?;
                let res = e.view_ols_res(ctx.as_ref())?;
                Ok(res)
            }
        }
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn into_arr_vec(self, ctx: Option<Context<'a>>) -> TpResult<Vec<ArrOk<'a>>> {
        match Arc::try_unwrap(self.0) {
            Ok(inner) => inner.into_inner().into_arr_vec(ctx),
            Err(expr) => {
                let mut e = expr.lock();
                e.eval_inplace(ctx.clone(), false)?;
                let arr_vec = e.view_arr_vec(ctx.as_ref())?;
                // need clone here
                let arr_vec = arr_vec
                    .into_iter()
                    .map(|a| match_arrok!(a, a, { a.view().to_owned().into() }))
                    .collect_trusted();
                Ok(arr_vec)
                // // safety: the expression has been evaluated
                // Ok(unsafe{std::mem::transmute(arr_view)})
            }
        }
    }

    #[inline]
    pub fn view_data(&self, context: Option<&Context<'a>>) -> TpResult<&Data<'a>> {
        let e = self.lock();
        let data = e.view_data(context)?;
        // safety: the array can only be read when the expression is already evaluated
        // so the data of the array should not be changed
        unsafe { Ok(std::mem::transmute(data)) }
    }

    #[inline]
    #[cfg(feature = "blas")]
    pub fn view_ols_res(&self, ctx: Option<&Context<'a>>) -> TpResult<Arc<OlsResult<'a>>> {
        let mut e = self.lock();
        e.eval_inplace(ctx.cloned(), false)?;
        let data = e.view_ols_res(ctx)?;
        Ok(data)
        // safety: the array can only be read when the expression is already evaluated
        // so the data of the array should not be changed
        // unsafe { Ok(std::mem::transmute(data)) }
    }

    #[inline]
    pub fn view_arr(&self, ctx: Option<&Context<'a>>) -> TpResult<&ArrOk<'a>> {
        let mut e = self.lock();
        e.eval_inplace(ctx.cloned(), false)?;
        let arr = e.view_arr(ctx)?;
        // safety: the array can only be read when the expression is already evaluated
        // so the data of the array should not be changed
        unsafe { Ok(std::mem::transmute(arr)) }
    }

    #[inline]
    pub fn view_arr_vec(&self, ctx: Option<&Context<'a>>) -> TpResult<Vec<&ArrOk<'a>>> {
        let mut e = self.lock();
        e.eval_inplace(ctx.cloned(), false)?;
        let arr = e.view_arr_vec(ctx)?;
        // safety: the array can only be read when the expression is already evaluated
        // so the data of the array should not be changed
        unsafe { Ok(std::mem::transmute(arr)) }
    }

    #[inline(always)]
    pub fn prepare(&self) {
        self.lock().prepare();
    }

    #[inline(always)]
    pub fn init_base_is_context(&self) -> bool {
        self.lock().init_base_is_context()
    }

    #[inline(always)]
    pub fn get_chain_base(&self) -> Data<'a> {
        self.lock().get_chain_base()
    }

    #[inline(always)]
    pub fn simplify_chain_nodes(&self, nodes: Vec<FuncNode<'a>>) -> Vec<FuncNode<'a>> {
        self.lock().simplify_chain_nodes(nodes)
    }

    #[inline(always)]
    pub fn collect_chain_nodes(&self, nodes: Vec<FuncNode<'a>>) -> Vec<FuncNode<'a>> {
        self.lock().collect_chain_nodes(nodes)
    }

    #[inline(always)]
    pub fn flatten(&self) -> Self {
        self.lock().flatten().into()
    }
}
