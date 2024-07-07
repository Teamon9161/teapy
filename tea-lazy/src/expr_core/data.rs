use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use super::{Expr, ExprElement, FuncNode};
#[cfg(feature = "blas")]
use crate::OlsResult;

use crate::{ColumnSelector, Context};
use derive_more::From;
use std::{fmt::Debug, sync::Arc};
use tea_core::prelude::*;

#[derive(From, Clone)]
pub enum Data<'a> {
    // an expression based on another expression
    Expr(Expr<'a>),
    // classical expression based on an array
    Arr(ArrOk<'a>),
    // an expression based on a vector of array
    ArrVec(Vec<ArrOk<'a>>),
    ArcArr(Arc<ArrOk<'a>>),
    // multi expressions share the same array
    // ArcArrVec(Vec<Arc<ArrOk<'a>>>),
    // an expression based on a context (e.g. groupby)
    Context(ColumnSelector<'a>),
    #[cfg(feature = "blas")]
    // only for least squares
    OlsRes(Arc<OlsResult<'a>>),
}

impl<'a> Debug for Data<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Data::Expr(expr) => write!(f, "{:?}", expr.lock()),
            Data::Arr(arr) => write!(f, "{arr:#?}"),
            Data::ArrVec(arr_vec) => {
                let mut out = f.debug_list();
                for arr in arr_vec {
                    out.entry(arr);
                }
                out.finish()
            }
            Data::ArcArr(arr) => write!(f, "{arr:#?}"),

            Data::Context(selector) => write!(f, "{selector:?}"),
            #[cfg(feature = "blas")]
            Data::OlsRes(res) => write!(f, "{res:#?}"),
            // #[cfg(feature = "arw")]
            // Data::Arrow(arr) => write!(f, "{arr:#?}"),
        }
    }
}

impl<'a> Default for Data<'a> {
    fn default() -> Self {
        Data::ArrVec(Vec::with_capacity(0))
    }
}

impl<'a> Data<'a> {
    #[inline]
    pub fn is_expr(&self) -> bool {
        matches!(&self, Data::Expr(_))
    }

    #[inline]
    pub fn is_context(&self) -> bool {
        matches!(&self, Data::Context(_))
    }

    #[inline]
    pub fn as_expr(&self) -> Option<&Expr<'a>> {
        match self {
            Data::Expr(expr) => Some(expr),
            _ => None,
        }
    }

    #[inline]
    pub fn as_expr_mut(&mut self) -> Option<&mut Expr<'a>> {
        match self {
            Data::Expr(expr) => Some(expr),
            _ => None,
        }
    }

    pub fn get_type(&self) -> &'static str {
        match self {
            Data::Expr(_) => "Expr",
            Data::Arr(arr) => arr.get_type(),
            Data::ArrVec(_) => "ArrVec",
            Data::ArcArr(_) => "ArcArr",

            Data::Context(_) => "Context",
            #[cfg(feature = "blas")]
            Data::OlsRes(_) => "OlsRes",
            // #[cfg(feature = "arw")]
            // Data::Arrow(_) => "Arrow",
        }
    }

    #[inline]
    pub fn dtype(&self) -> String {
        match self {
            Data::Expr(e) => e.dtype(),
            Data::Arr(arr) => format!("{:?}", arr.dtype()),
            _ => "Unknown".to_string(),
        }
    }

    #[inline]
    pub fn is_owned(&self) -> bool {
        match self {
            Data::Expr(expr) => expr.is_owned(),
            Data::Arr(arr) => arr.is_owned(),
            Data::ArrVec(_) => false,
            Data::ArcArr(_) => false,

            Data::Context(_) => false,
            #[cfg(feature = "blas")]
            Data::OlsRes(_) => false,
            // currently we can only read arrow array from file
            // so we assume that the array is owned
            // #[cfg(feature = "arw")]
            // Data::Arrow(_) => true,
        }
    }

    #[inline]
    pub fn prepare(&mut self) {
        match self {
            Data::Arr(arr) => arr.prepare(),
            Data::ArrVec(arr_vec) => arr_vec.par_iter_mut().for_each(|arr| arr.prepare()),
            Data::Expr(expr) => expr.prepare(),
            _ => {}
        }
    }

    #[inline]
    pub fn init_base_is_context(&self) -> bool {
        match self {
            Data::Expr(expr) => expr.init_base_is_context(),

            Data::Context(_) => true,
            _ => false,
        }
    }

    // get initial base of the expression
    #[inline]
    pub fn get_chain_base(&self) -> Data<'a> {
        match self {
            Data::Expr(expr) => expr.get_chain_base(),
            _ => self.clone(),
        }
    }

    #[inline]
    pub fn simplify_base(&mut self) {
        if let Data::Expr(expr) = self {
            if expr.strong_count() == 1 {
                let base_expr = std::mem::take(expr);
                let mut base_expr = base_expr.into_inner().unwrap().base;
                base_expr.simplify_base();
                *self = base_expr;
            } else {
                expr.simplify();
            }
        }
    }

    #[inline]
    pub fn simplify_chain_nodes(&self, nodes: Vec<FuncNode<'a>>) -> Vec<FuncNode<'a>> {
        match self {
            Data::Expr(expr) => {
                if expr.strong_count() == 1 {
                    expr.simplify_chain_nodes(nodes)
                } else {
                    nodes
                }
            }
            _ => nodes,
        }
    }

    // get the all nodes of the expression
    #[inline(always)]
    pub fn collect_chain_nodes(&self, nodes: Vec<FuncNode<'a>>) -> Vec<FuncNode<'a>> {
        match self {
            Data::Expr(expr) => expr.collect_chain_nodes(nodes),
            _ => nodes,
        }
    }

    #[inline]
    pub fn context_clone(&self) -> Option<Self> {
        match self {
            Data::Expr(expr) => Some(expr.context_clone().into()),
            Data::Context(cs) => Some(cs.clone().into()),
            _ => None,
        }
    }

    #[inline]
    pub fn into_arr(self, ctx: Option<Context<'a>>) -> TResult<ArrOk<'a>> {
        match self {
            Data::Arr(arr) => Ok(arr),
            Data::Expr(e) => e.into_arr(ctx),

            Data::Context(col) => {
                let ctx1 = ctx
                    .clone()
                    .ok_or_else(|| terr!("The context is not provided"))?;
                let out = ctx1.get(col.clone())?;
                // need clone here
                Ok(out.into_expr()?.view_arr(None)?.deref().into_owned())
            }
            // #[cfg(feature = "arw")]
            // Data::Arrow(arr) => Ok(ArrOk::from_arrow(arr)),
            _ => tbail!("The output of the expression is not an array"),
        }
    }

    #[inline]
    pub fn into_arr_vec(self, ctx: Option<Context<'a>>) -> TResult<Vec<ArrOk<'a>>> {
        match self {
            Data::ArrVec(arr) => Ok(arr),
            Data::Expr(e) => e.into_arr_vec(ctx),

            Data::Context(col) => {
                let ctx1 = ctx
                    .clone()
                    .ok_or_else(|| terr!("The context is not provided"))?;
                let out = ctx1.get(col.clone())?;
                // need clone here
                Ok(out
                    .into_expr()?
                    .view_arr_vec(None)?
                    .into_iter()
                    .map(|a| a.deref().into_owned())
                    .collect_trusted_to_vec())
            }
            _ => tbail!("The output of the expression is not an array vector"),
        }
    }

    #[cfg(feature = "blas")]
    #[inline]
    pub fn into_ols_res(self, ctx: Option<Context<'a>>) -> TResult<Arc<OlsResult<'a>>> {
        match self {
            Data::OlsRes(res) => Ok(res),
            Data::Expr(e) => e.into_ols_res(ctx),
            Data::Context(col) => {
                let ctx1 = ctx
                    .clone()
                    .ok_or_else(|| terr!("The context is not provided"))?;
                let out = ctx1.get(col.clone())?;
                Ok(out.into_expr()?.view_ols_res(None)?)
            }
            _ => tbail!(
                "The output of the expression is not an OlsResult: {:?}",
                &self
            ),
        }
    }

    #[cfg(feature = "blas")]
    #[inline]
    pub fn view_ols_res<'b>(&'b self, ctx: Option<&'b Context<'a>>) -> TResult<Arc<OlsResult<'a>>> {
        match self {
            Data::OlsRes(res) => Ok(res.clone()),
            Data::Expr(e) => e.view_ols_res(ctx),
            Data::Context(col) => {
                let out = ctx
                    .ok_or_else(|| terr!("The context is not provided"))?
                    .get(col.clone())?
                    .into_expr()?;
                out.view_ols_res(None)
            }
            _ => tbail!("The output of the expression is not an OlsResult"),
        }
    }

    #[inline]
    pub fn view_arr<'b>(&'b self, ctx: Option<&'b Context<'a>>) -> TResult<&'b ArrOk<'a>> {
        match self {
            Data::Arr(arr) => Ok(arr),
            Data::Expr(e) => e.view_arr(ctx),

            Data::Context(col) => {
                let out = ctx
                    .ok_or_else(|| terr!("The context is not provided"))?
                    .get(col.clone())?
                    .into_expr()?;
                out.view_arr(None)
            }
            _ => tbail!("The output of the expression is not an array, {:?}", self),
        }
    }

    #[inline]
    pub fn view_arr_vec<'b>(&'b self, ctx: Option<&'b Context<'a>>) -> TResult<Vec<&'b ArrOk<'a>>> {
        match self {
            Data::ArrVec(arr_vec) => Ok(arr_vec.iter().collect::<Vec<_>>()),
            Data::Expr(e) => e.view_arr_vec(ctx),

            Data::Context(col) => {
                let ctx = ctx.ok_or_else(|| terr!("The context is not provided"))?;
                let out = ctx.get(col.clone())?;
                Ok(out
                    .into_exprs()
                    .iter()
                    .map(|e| e.view_arr(None).unwrap())
                    .collect_trusted_to_vec())
            }
            _ => tbail!("The output of the expression is not an array"),
        }
    }
}

impl<'a, T: ExprElement + 'a> From<T> for Data<'a>
where
    ArbArray<'a, T>: Into<Data<'a>>,
{
    #[inline(always)]
    fn from(t: T) -> Self {
        let a = ArbArray::from_scalar(t);
        a.into()
    }
}

impl<'a, T: 'a> From<ArrD<T>> for Data<'a>
where
    ArrD<T>: Into<ArbArray<'a, T>>,
    ArrOk<'a>: From<ArbArray<'a, T>>,
{
    #[inline]
    fn from(arr: ArrD<T>) -> Self {
        let arb: ArbArray<'a, T> = arr.into();
        Data::Arr(arb.into())
    }
}

impl<'a, T: 'a> From<ArbArray<'a, T>> for Data<'a>
where
    ArrOk<'a>: From<ArbArray<'a, T>>,
{
    #[inline]
    fn from(arb: ArbArray<'a, T>) -> Self {
        Data::Arr(arb.into())
    }
}

impl<'a, T: 'a> From<ArrViewD<'a, T>> for Data<'a>
where
    ArrOk<'a>: From<ArbArray<'a, T>>,
{
    #[inline(always)]
    fn from(arr: ArrViewD<'a, T>) -> Self {
        let a: ArbArray<'a, T> = arr.into();
        Data::Arr(a.into())
    }
}

impl<'a, T: 'a> From<ArrViewMutD<'a, T>> for Data<'a>
where
    ArrOk<'a>: From<ArbArray<'a, T>>,
{
    #[inline]
    fn from(arr: ArrViewMutD<'a, T>) -> Self {
        let arb: ArbArray<'a, T> = arr.into();
        Data::Arr(arb.into())
    }
}

#[cfg(feature = "blas")]
impl<'a> From<OlsResult<'a>> for Data<'a> {
    #[inline(always)]
    fn from(res: OlsResult<'a>) -> Self {
        Data::OlsRes(Arc::new(res))
    }
}
