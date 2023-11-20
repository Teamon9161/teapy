#[macro_export]
macro_rules! auto_impl_view {
    (in1, $func:ident, ($($p:ident: $p_ty:ty),* $(,)?)) => {
        fn $func(&mut self $(, $p: $p_ty)*) -> &mut Self {
            self.chain_f_ctx(move |(data, ctx)| {
                let arr = data.view_arr(ctx.as_ref())?;
                match_arrok!(numeric arr, a, { Ok((a.view().$func($($p),*).into(), ctx)) })
            });
            self
        }
    };
    (in2, $func:ident, ($($p:ident: $p_ty:ty),* $(,)?)) => {
        pub fn $func(&mut self, other: Expr<'a>, $($p: $p_ty),*) -> &mut Self {
            self.chain_f_ctx(move |(data, ctx): FuncOut<'a>| {
                let other_arr = other.view_arr(ctx.as_ref())?;
                let arr = data.view_arr(ctx.as_ref())?;
                match_arrok!(
                    (arr, a, F64, F32, I64, I32),
                    (other_arr, o, F64, F32, I64, I32),
                    {
                        Ok((a.view().$func(&o.view(), $($p),*).into(), ctx))
                    }
                )
            });
            self
        }
    };
}

#[macro_export]
macro_rules! auto_impl_f64_func {
    ($func:ident, ($($p:ident: $p_ty:ty),* $(,)?)) => {
        pub fn $func(&mut self $(, $p: $p_ty)*) -> &mut Self{
            self.chain_f_ctx(move |(data, ctx)| {
                let arr = data.view_arr(ctx.as_ref())?;
                match_arrok!(numeric arr, a, { Ok((a.view().map(|v| v.f64().$func($($p),*)).into(), ctx)) })
            });
            self
        }
    };
}

#[macro_export]
macro_rules! auto_impl_viewmut {
    (in1, $func:ident, ($($p:ident: $p_ty:ty),* $(,)?)) => {
        pub fn $func(&mut self, $($p: $p_ty),*) -> &mut Self {
            self.chain_f_ctx(move |(data, ctx)| {
                let mut arr = data.into_arr(ctx.clone())?;
                match_arrok!(numeric &mut arr, a, {
                    a.view_mut().$func($($p),*);
                });
                Ok((arr.into(), ctx))
            });
            self
        }
    }
}
