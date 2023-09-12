use super::Data;
use crate::{ArbArray, ArrD, ArrOk, Context, ExprElement, OlsResult, TpResult};
use std::{fmt::Debug, sync::Arc};

#[derive(Default)]
pub struct ExprInner<'a> {
    pub base: Data<'a>,
    pub name: Option<String>,
    pub nodes: Vec<FuncNode<'a>>,
    // a field to store the output,
    // as the output of different context is different
    // we can not change expression base in each context
    pub cxt_ref: Option<Data<'a>>,
}

impl<'a, T: ExprElement + 'a> From<T> for ExprInner<'a> {
    fn from(arr: T) -> Self {
        let a: ArbArray<'a, T> = arr.into();
        ExprInner {
            base: Data::Arr(a.into()),
            name: None,
            nodes: Vec::new(),
            cxt_ref: None,
        }
    }
}

impl<'a, T: ExprElement + 'a> From<ArrD<T>> for ExprInner<'a> {
    fn from(arr: ArrD<T>) -> Self {
        let a: ArbArray<'a, T> = arr.into();
        ExprInner {
            base: Data::Arr(a.into()),
            name: None,
            nodes: Vec::new(),
            cxt_ref: None,
        }
    }
}

impl<'a, T: ExprElement + 'a> From<ArbArray<'a, T>> for ExprInner<'a> {
    fn from(arr: ArbArray<'a, T>) -> Self {
        ExprInner {
            base: Data::Arr(arr.into()),
            name: None,
            nodes: Vec::new(),
            cxt_ref: None,
        }
    }
}

// pub type ExprBase<'a> = Arc<Mutex<Data<'a>>>;
pub type FuncOut<'a> = (Data<'a>, Option<Context<'a>>);
pub type FuncNode<'a> = Arc<dyn Fn(FuncOut<'a>) -> TpResult<FuncOut<'a>> + Send + Sync + 'a>;

impl<'a> Debug for ExprInner<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.step() == 0 {
            writeln!(f, "{:?}", self.base)?;
            Ok(())
        } else {
            // if let Data::Expr(e) = &self.base {
            //     let out = format!("{:?}", e.lock());
            //     writeln!(f, "{}", out.split("Expr").next().unwrap())?;
            // }
            let mut f = f.debug_struct("Expr");
            if let Some(name) = &self.name {
                f.field("name", name);
            }
            f.field("step", &self.step_acc()).finish()
        }
    }
}

impl<'a> ExprInner<'a> {
    pub fn new(data: Data<'a>, name: Option<String>) -> Self {
        ExprInner {
            base: data,
            name,
            nodes: Vec::new(),
            cxt_ref: None,
        }
    }

    pub fn step(&self) -> usize {
        self.nodes.len()
    }

    pub fn context_clone(&self) -> Self {
        // self.flatten()
        let new_nodes = self.nodes.clone();
        let new_base = self.base.context_clone();
        let name = self.name.clone();
        ExprInner {
            base: new_base.unwrap(),
            name,
            nodes: new_nodes,
            cxt_ref: None,
        }
    }

    pub fn step_acc(&self) -> usize {
        let self_step = self.step();
        if let Some(expr) = self.base.as_expr() {
            self_step + expr.step_acc()
        } else {
            self_step
        }
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn name_owned(&self) -> Option<String> {
        self.name.clone()
    }

    pub fn set_name(&mut self, name: Option<String>) {
        self.name = name;
    }

    pub fn set_base(&mut self, data: Data<'a>) {
        self.base = data;
    }

    pub fn is_owned(&self) -> bool {
        if self.step() == 0 {
            self.base.is_owned()
        } else {
            false
        }
    }

    pub fn base_type(&self) -> &'static str {
        self.base.get_type()
    }

    pub fn dtype(&self) -> String {
        if self.step() == 0 {
            self.base.dtype()
        } else {
            "Unknown".to_string()
        }
    }

    /// chain a new function to current function chain
    pub fn chain_f_ctx<F>(&mut self, f: F)
    where
        F: Fn(FuncOut<'a>) -> TpResult<FuncOut<'a>> + Send + Sync + 'a,
    {
        self.nodes.push(Arc::new(f));
    }

    pub fn eval_inplace(&mut self, mut ctx: Option<Context<'a>>) -> TpResult<&mut Self> {
        if self.step() == 0 {
            if let Some(e) = self.base.as_expr_mut() {
                e.eval_inplace(ctx)?;
            }
            return Ok(self);
        }

        if ctx.is_none() {
            let mut data = std::mem::take(&mut self.base);
            for f in &self.nodes {
                (data, ctx) = f((data, ctx))?;
            }
            // do not clear the nodes if evaluate in context
            // as the result would be different in different context
            self.nodes.clear();
            self.base = data;
        } else {
            let mut data = self.base.get_chain_base();
            let nodes = self.collect_chain_nodes(vec![]);
            self.base = data.clone();
            self.nodes = nodes;
            // dbg!("inner eval inplace once, step: {:?}", self.nodes.len());
            for f in &self.nodes {
                (data, ctx) = f((data, ctx))?;
            }
            self.cxt_ref = Some(data);
        }
        Ok(self)
    }

    pub fn eval(mut self, ctx: Option<Context<'a>>) -> TpResult<Self> {
        self.eval_inplace(ctx)?;
        Ok(self)
    }

    pub fn into_out(self, mut ctx: Option<Context<'a>>) -> TpResult<Data<'a>> {
        let mut data = self.base;
        for f in self.nodes.into_iter() {
            (data, ctx) = f((data, ctx))?;
        }
        Ok(data)
    }

    pub fn into_arr(self, ctx: Option<Context<'a>>) -> TpResult<ArrOk<'a>> {
        self.into_out(ctx.clone())
            .map(|data| data.into_arr(ctx).unwrap())
    }

    pub fn into_arr_vec(self, ctx: Option<Context<'a>>) -> TpResult<Vec<ArrOk<'a>>> {
        self.into_out(ctx.clone())
            .map(|data| data.into_arr_vec(ctx).unwrap())
    }

    #[cfg(feature = "blas")]
    pub fn into_ols_res(self, ctx: Option<Context<'a>>) -> TpResult<Arc<OlsResult<'a>>> {
        self.into_out(ctx.clone())
            .map(|data| data.into_ols_res(ctx).unwrap())
    }

    pub fn view_arr<'b>(&'b self, ctx: Option<&'b Context<'a>>) -> TpResult<&'b ArrOk<'a>> {
        if (self.step() > 0) & ctx.is_none() {
            return Err("Can not view array before evaluate the expression".into());
        }
        if ctx.is_some() && self.step() != 0 {
            self.cxt_ref.as_ref().unwrap().view_arr(ctx)
        } else {
            self.base.view_arr(ctx)
        }
    }

    pub fn view_arr_vec<'b>(
        &'b self,
        ctx: Option<&'b Context<'a>>,
    ) -> TpResult<Vec<&'b ArrOk<'a>>> {
        if (self.step() > 0) & ctx.is_none() {
            return Err("Can not view array before evaluate the expression".into());
        }
        if ctx.is_some() {
            if self.step() == 0 {
                return self.base.view_arr_vec(ctx);
            }
            self.cxt_ref.as_ref().unwrap().view_arr_vec(ctx)
        } else {
            self.base.view_arr_vec(ctx)
        }
    }

    #[cfg(feature = "blas")]
    pub fn view_ols_res<'b>(
        &'b self,
        ctx: Option<&'b Context<'a>>,
    ) -> TpResult<Arc<OlsResult<'a>>> {
        if (self.step() > 0) & ctx.is_none() {
            return Err("Do not lock base before evaluate the expression".into());
        }
        if ctx.is_some() {
            if self.step() == 0 {
                return self.base.view_ols_res(ctx);
            }
            self.cxt_ref.as_ref().unwrap().view_ols_res(ctx)
        } else {
            self.base.view_ols_res(ctx)
        }
    }

    pub fn view_data(&self) -> TpResult<&Data<'a>> {
        if self.step() > 0 {
            return Err("Do not lock base before evaluate the expression".into());
        }
        Ok(&self.base)
    }

    pub fn get_chain_base(&self) -> Data<'a> {
        self.base.get_chain_base()
    }

    pub fn collect_chain_nodes(&self, nodes: Vec<FuncNode<'a>>) -> Vec<FuncNode<'a>> {
        if !self.nodes.is_empty() {
            let mut out = self.nodes.clone();
            out.extend(nodes);
            self.base.collect_chain_nodes(out)
        } else {
            self.base.collect_chain_nodes(nodes)
        }
    }

    pub fn flatten(&self) -> Self {
        let base = self.get_chain_base();
        let nodes = self.collect_chain_nodes(Vec::new());
        Self {
            base,
            name: self.name.clone(),
            nodes,
            cxt_ref: None,
        }
    }
}
