use super::{Expr, ExprElement, FuncNode};
#[cfg(feature = "blas")]
use crate::lazy::OlsResult;
use crate::{lazy::ColumnSelector, ArrViewD, CollectTrustedToVec};
use crate::{ArbArray, ArrD, ArrOk, ArrViewMutD, Context, GetDataType, GetNone, TpResult};
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Clone)]
pub enum Data<'a> {
    Expr(Expr<'a>),         // an expression based on another expression
    Arr(ArrOk<'a>),         // classical expression based on an array
    ArrVec(Vec<ArrOk<'a>>), // an expression based on a vector of array
    ArcArr(Arc<ArrOk<'a>>),
    // ArcArrVec(Vec<Arc<ArrOk<'a>>>)   ,        // multi expressions share the same array
    Context(ColumnSelector<'a>), // an expression based on a context (e.g. groupby
    #[cfg(feature = "blas")]
    OlsRes(Arc<OlsResult<'a>>), // only for least squares
                                 // #[cfg(feature = "arw")]
                                 // Arrow(Arc<dyn Array>),
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
            Data::Context(selector) => write!(f, "{selector:?}"),
            Data::ArcArr(arr) => write!(f, "{arr:#?}"),
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
    pub fn is_expr(&self) -> bool {
        matches!(&self, Data::Expr(_))
    }

    pub fn is_context(&self) -> bool {
        matches!(&self, Data::Context(_))
    }

    pub fn as_expr(&self) -> Option<&Expr<'a>> {
        match self {
            Data::Expr(expr) => Some(expr),
            _ => None,
        }
    }

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

    pub fn dtype(&self) -> String {
        match self {
            Data::Expr(e) => e.dtype(),
            Data::Arr(arr) => format!("{:?}", arr.dtype()),
            _ => "Unknown".to_string(),
        }
    }

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

    pub fn init_base_is_context(&self) -> bool {
        match self {
            Data::Expr(expr) => expr.init_base_is_context(),
            Data::Context(_) => true,
            _ => false,
        }
    }

    // get initial base of the expression
    pub fn get_chain_base(&self) -> Data<'a> {
        match self {
            Data::Expr(expr) => expr.get_chain_base(),
            _ => self.clone(),
        }
    }

    // get the all nodes of the expression
    pub fn collect_chain_nodes(&self, nodes: Vec<FuncNode<'a>>) -> Vec<FuncNode<'a>> {
        match self {
            Data::Expr(expr) => expr.collect_chain_nodes(nodes),
            _ => nodes,
        }
    }

    pub fn context_clone(&self) -> Option<Self> {
        match self {
            Data::Expr(expr) => Some(expr.context_clone().into()),
            Data::Context(cs) => Some(cs.clone().into()),
            _ => None,
        }
    }

    pub fn into_arr(self, ctx: Option<Context<'a>>) -> TpResult<ArrOk<'a>> {
        match self {
            Data::Arr(arr) => Ok(arr),
            Data::Expr(e) => e.into_arr(ctx),
            Data::Context(col) => {
                let ctx1 = ctx.clone().ok_or("The context is not provided")?;
                let out = ctx1.get(col.clone())?;
                // need clone here
                Ok(out.into_expr()?.view_arr(None)?.deref().into_owned())
            }
            // #[cfg(feature = "arw")]
            // Data::Arrow(arr) => Ok(ArrOk::from_arrow(arr)),
            _ => Err("The output of the expression is not an array".into()),
        }
    }

    pub fn into_arr_vec(self, ctx: Option<Context<'a>>) -> TpResult<Vec<ArrOk<'a>>> {
        match self {
            Data::ArrVec(arr) => Ok(arr),
            Data::Expr(e) => e.into_arr_vec(ctx),
            Data::Context(col) => {
                let ctx1 = ctx.clone().ok_or("The context is not provided")?;
                let out = ctx1.get(col.clone())?;
                // need clone here
                Ok(out
                    .into_expr()?
                    .view_arr_vec(None)?
                    .into_iter()
                    .map(|a| a.deref().into_owned())
                    .collect_trusted())
            }
            _ => Err("The output of the expression is not an array vector".into()),
        }
    }

    #[cfg(feature = "blas")]
    pub fn into_ols_res(self, ctx: Option<Context<'a>>) -> TpResult<Arc<OlsResult<'a>>> {
        match self {
            Data::OlsRes(res) => Ok(res),
            Data::Expr(e) => e.into_ols_res(ctx),
            Data::Context(col) => {
                let ctx1 = ctx.clone().ok_or("The context is not provided")?;
                let out = ctx1.get(col.clone())?;
                Ok(out.into_expr()?.view_ols_res(None)?)
            }
            _ => Err(format!(
                "The output of the expression is not an OlsResult: {:?}",
                &self
            )
            .into()),
        }
    }

    #[cfg(feature = "blas")]
    pub fn view_ols_res<'b>(
        &'b self,
        ctx: Option<&'b Context<'a>>,
    ) -> TpResult<Arc<OlsResult<'a>>> {
        match self {
            Data::OlsRes(res) => Ok(res.clone()),
            Data::Expr(e) => e.view_ols_res(ctx),
            Data::Context(col) => {
                let out = ctx
                    .ok_or("The context is not provided")?
                    .get(col.clone())?
                    .into_expr()?;
                out.view_ols_res(None)
            }
            _ => Err("The output of the expression is not an OlsResult".into()),
        }
    }

    pub fn view_arr<'b>(&'b self, ctx: Option<&'b Context<'a>>) -> TpResult<&'b ArrOk<'a>> {
        match self {
            Data::Arr(arr) => Ok(arr),
            Data::Expr(e) => e.view_arr(ctx),
            Data::Context(col) => {
                let out = ctx
                    .ok_or("The context is not provided")?
                    .get(col.clone())?
                    .into_expr()?;
                out.view_arr(None)
            }
            _ => Err(format!("The output of the expression is not an array, {:?}", self).into()),
        }
    }

    pub fn view_arr_vec<'b>(
        &'b self,
        ctx: Option<&'b Context<'a>>,
    ) -> TpResult<Vec<&'b ArrOk<'a>>> {
        match self {
            Data::ArrVec(arr_vec) => Ok(arr_vec.iter().collect::<Vec<_>>()),
            Data::Expr(e) => e.view_arr_vec(ctx),
            Data::Context(col) => {
                let ctx = ctx.ok_or("The context is not provided")?;
                let out = ctx.get(col.clone())?;
                Ok(out
                    .into_exprs()
                    .iter()
                    .map(|e| e.view_arr(None).unwrap())
                    .collect_trusted())
            }
            _ => Err("The output of the expression is not an array".into()),
        }
    }
}

impl<'a> From<ArrOk<'a>> for Data<'a> {
    fn from(arr: ArrOk<'a>) -> Self {
        Data::Arr(arr)
    }
}

impl<'a> From<Expr<'a>> for Data<'a> {
    fn from(expr: Expr<'a>) -> Self {
        Data::Expr(expr)
    }
}

impl<'a, T: ExprElement + 'a> From<T> for Data<'a> {
    fn from(t: T) -> Self {
        let a: ArbArray<'a, T> = t.into();
        a.into()
    }
}

impl<'a, T> From<Option<T>> for Data<'a>
where
    T: GetNone + ExprElement + 'a,
{
    fn from(v: Option<T>) -> Self {
        let v = v.unwrap_or_else(T::none);
        v.into()
    }
}

impl<'a, T: GetDataType + 'a> From<ArrD<T>> for Data<'a> {
    #[inline]
    fn from(arr: ArrD<T>) -> Self {
        let arb: ArbArray<'a, T> = arr.into();
        Data::Arr(arb.into())
    }
}

impl<'a, T: GetDataType + 'a> From<ArbArray<'a, T>> for Data<'a> {
    #[inline]
    fn from(arb: ArbArray<'a, T>) -> Self {
        Data::Arr(arb.into())
    }
}

impl<'a, T: GetDataType + 'a> From<ArrViewD<'a, T>> for Data<'a> {
    fn from(arr: ArrViewD<'a, T>) -> Self {
        let a: ArbArray<'a, T> = arr.into();
        Data::Arr(a.into())
    }
}

impl<'a, T: GetDataType + 'a> From<ArrViewMutD<'a, T>> for Data<'a> {
    #[inline]
    fn from(arr: ArrViewMutD<'a, T>) -> Self {
        let arb: ArbArray<'a, T> = arr.into();
        Data::Arr(arb.into())
    }
}

impl<'a> From<Vec<ArrOk<'a>>> for Data<'a> {
    fn from(arr_vec: Vec<ArrOk<'a>>) -> Self {
        Data::ArrVec(arr_vec)
    }
}

impl<'a> From<Arc<ArrOk<'a>>> for Data<'a> {
    fn from(arr: Arc<ArrOk<'a>>) -> Self {
        Data::ArcArr(arr)
    }
}

impl<'a> From<ColumnSelector<'a>> for Data<'a> {
    fn from(col: ColumnSelector<'a>) -> Self {
        Data::Context(col)
    }
}

#[cfg(feature = "blas")]
impl<'a> From<OlsResult<'a>> for Data<'a> {
    fn from(res: OlsResult<'a>) -> Self {
        Data::OlsRes(Arc::new(res))
    }
}
