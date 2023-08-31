// use crate::{Cast, OptUsize, PyValue, TpResult, Exprs};
use parking_lot::Mutex;
use parking_lot::MutexGuard;
// // use rayon::prelude::*;
use std::{fmt::Debug, marker::PhantomData, mem, sync::Arc};

// #[cfg(feature = "option_dtype")]
// use crate::{OptF32, OptF64, OptI32, OptI64};
use super::expr_element::ExprElement;
use crate::lazy::ColumnSelector;
use crate::lazy::OlsResult;
use crate::ArrViewMut;
use crate::ArrViewMutD;
use crate::{ArbArray, ArrD, ArrOk, ArrViewD, Context, GetDataType, TpResult};

pub struct Expr<'a> {
    pub base: ExprBase<'a>,
    pub name: Option<String>,
    pub nodes: Vec<FuncNode<'a>>,
}

pub type ExprBase<'a> = Arc<Mutex<Data<'a>>>;
pub type FuncOut<'a> = (Data<'a>, Option<Context<'a>>);
pub type FuncNode<'a> = Box<dyn FnOnce(FuncOut<'a>) -> TpResult<FuncOut<'a>> + Send + Sync + 'a>;

impl<'a> Clone for Expr<'a> {
    fn clone(&self) -> Self {
        let ref_expr = Expr {
            base: self.base.clone(),
            name: self.name.clone(),
            nodes: Vec::new(),
        };
        Expr::new(ref_expr.into(), self.name.clone())
    }
}

pub enum Data<'a> {
    Expr(Expr<'a>),         // an expression based on another expression
    Arr(ArrOk<'a>),         // classical expression based on an array
    ArrVec(Vec<ArrOk<'a>>), // an expression based on a vector of array
    ArcArr(Arc<ArrOk<'a>>),
    // ArcArrVec(Vec<Arc<ArrOk<'a>>>)   ,        // multi expressions share the same array
    Context(ColumnSelector<'a>), // an expression based on a context (e.g. groupby
    #[cfg(feature = "blas")]
    OlsRes(Arc<OlsResult<'a>>), // only for least squares
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
        }
    }

    pub fn into_arr(self, ctx: Option<Context<'a>>) -> TpResult<(ArrOk<'a>, Option<Context<'a>>)> {
        match self {
            Data::Arr(arr) => Ok((arr, ctx)),
            Data::Expr(e) => e.into_arr(ctx),
            _ => Err("The output of the expression is not an array".into()),
        }
    }

    pub fn view_arr(&self) -> TpResult<&ArrOk<'a>> {
        match self {
            Data::Arr(arr) => Ok(&arr),
            Data::Expr(e) => e.lock_base().view_arr(),
            _ => Err("The output of the expression is not an array".into()),
        }
    }

    // pub fn view_arr(&mut self, ctx: Option<Context<'a>>) -> TpResult<(&mut ArrOk<'a>, Option<Context<'a>>)> {
    //     match self {
    //         Data::Arr(arr) => Ok((arr, ctx)),
    //         Data::Expr(e) => e.view_arr(ctx),
    //         _ => Err("The output of the expression is not an array".into())
    //     }
    // }
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

impl<'a> Expr<'a> {
    pub fn new(data: Data<'a>, name: Option<String>) -> Self {
        Expr {
            base: Arc::new(Mutex::new(data)),
            name,
            nodes: Vec::new(),
        }
    }

    pub fn lock_base(&self) -> MutexGuard<'a, Data<'a>> {
        if self.step() > 0 {
            panic!("Do not lock base before evaluate the expression")
        }
        self.base.lock()
    }

    pub fn step(&self) -> usize {
        self.nodes.len()
    }

    pub fn step_acc(&self) -> usize {
        let self_step = self.step();
        if let Some(expr) = self.base.lock().as_expr() {
            self_step + expr.step_acc()
        } else {
            self_step
        }
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_ref().map(|s| s.as_str())
    }

    pub fn name_owned(&self) -> Option<String> {
        self.name.clone()
    }

    pub fn set_name(&mut self, name: Option<String>) {
        self.name = name;
    }

    pub fn base_type(&self) -> &'static str {
        &self.base.lock().get_type()
    }

    /// chain a new function to current function chain
    pub fn chain_f_ctx<F>(&mut self, f: F)
    where
        F: FnOnce(FuncOut<'a>) -> TpResult<FuncOut<'a>> + Send + Sync + 'a,
    {
        self.nodes.push(Box::new(f));
    }

    // chain a function without context
    pub fn chain_f<F>(&mut self, f: F)
    where
        F: FnOnce(Data<'a>) -> TpResult<Data<'a>> + Send + Sync + 'a,
    {
        let f = Box::new(move |(data, ctx)| Ok((f(data)?, ctx)));
        self.nodes.push(f);
    }

    pub fn eval_inplace(&mut self, mut ctx: Option<Context<'a>>) -> TpResult<Option<Context<'a>>> {
        let mut data = if let Some(base) = Arc::get_mut(&mut self.base) {
            std::mem::take(base).into_inner()
        } else {
            Data::Expr(Expr {
                base: self.base.clone(),
                name: self.name.clone(),
                nodes: Vec::new(),
            })
        };
        let nodes = std::mem::take(&mut self.nodes);
        for f in nodes.into_iter() {
            (data, ctx) = f((data, ctx))?;
        }
        self.base = Arc::new(Mutex::new(data));
        Ok(ctx)
    }

    pub fn eval(mut self, ctx: Option<Context<'a>>) -> TpResult<(Self, Option<Context<'a>>)> {
        let ctx = self.eval_inplace(ctx)?;
        Ok((self, ctx))
    }

    pub fn into_out(self, ctx: Option<Context<'a>>) -> TpResult<(Data<'a>, Option<Context<'a>>)> {
        let (expr, ctx) = self.eval(ctx)?;
        if let Ok(base) = Arc::try_unwrap(expr.base) {
            Ok((base.into_inner(), ctx))
        } else {
            unreachable!("the new expression of the evaluation result should not be shared")
        }
    }

    pub fn into_arr(self, ctx: Option<Context<'a>>) -> TpResult<(ArrOk<'a>, Option<Context<'a>>)> {
        self.into_out(ctx)
            .map(|(data, ctx)| (data.into_arr(ctx).unwrap()))
    }

    pub fn view_arr(&self) -> TpResult<ArrOk<'_>> {
        if self.step() > 0 {
            panic!("Do not lock base before evaluate the expression")
        }
        self.lock_base().view_arr()
    }
}
