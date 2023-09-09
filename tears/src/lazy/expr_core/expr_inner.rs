use super::Data;
use crate::{ArbArray, ArrD, ArrOk, Context, ExprElement, TpResult};
use std::{fmt::Debug, sync::Arc};

#[derive(Default)]
pub struct ExprInner<'a> {
    pub base: Data<'a>,
    pub name: Option<String>,
    pub nodes: Vec<FuncNode<'a>>,
}

impl<'a, T: ExprElement + 'a> From<T> for ExprInner<'a> {
    fn from(arr: T) -> Self {
        let a: ArbArray<'a, T> = arr.into();
        ExprInner {
            base: Data::Arr(a.into()),
            name: None,
            nodes: Vec::new(),
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
        }
    }
}

impl<'a, T: ExprElement + 'a> From<ArbArray<'a, T>> for ExprInner<'a> {
    fn from(arr: ArbArray<'a, T>) -> Self {
        ExprInner {
            base: Data::Arr(arr.into()),
            name: None,
            nodes: Vec::new(),
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
        }
    }

    pub fn step(&self) -> usize {
        self.nodes.len()
    }

    pub fn context_clone(&self) -> Self {
        let new_nodes = self.nodes.clone();
        let new_base = self.base.context_clone();
        let name = self.name.clone();
        ExprInner {
            base: new_base.unwrap(),
            name,
            nodes: new_nodes,
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

    // chain a function without context
    pub fn chain_f<F>(&mut self, f: F)
    where
        F: Fn(Data<'a>) -> TpResult<Data<'a>> + Send + Sync + 'a,
    {
        let f = Arc::new(move |(data, ctx)| Ok((f(data)?, ctx)));
        self.nodes.push(f);
    }

    pub fn eval_inplace(&mut self, mut ctx: Option<Context<'a>>) -> TpResult<&mut Self> {
        if self.step() == 0 {
            if let Some(e) = self.base.as_expr_mut() {
                e.eval_inplace(ctx)?;
            }
            return Ok(self);
        }
        let mut data = std::mem::take(&mut self.base);
        // let nodes = std::mem::take(&mut self.nodes);
        for f in &self.nodes {
            (data, ctx) = f((data, ctx))?;
        }
        self.base = data;
        // if ctx.is_none() {
        self.nodes.clear();
        // }
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

    pub fn view_arr<'b>(&'b self, ctx: Option<&'b Context<'a>>) -> TpResult<&'b ArrOk<'a>> {
        if (self.step() > 0) & ctx.is_none() {
            return Err("Do not lock base before evaluate the expression".into());
        }
        self.base.view_arr(ctx)
    }

    pub fn view_arr_vec<'b>(
        &'b self,
        ctx: Option<&'b Context<'a>>,
    ) -> TpResult<Vec<&'b ArrOk<'a>>> {
        if (self.step() > 0) & ctx.is_none() {
            return Err("Do not lock base before evaluate the expression".into());
        }
        self.base.view_arr_vec(ctx)
    }

    pub fn view_data(&self) -> TpResult<&Data<'a>> {
        if self.step() > 0 {
            return Err("Do not lock base before evaluate the expression".into());
        }
        Ok(&self.base)
    }
}
