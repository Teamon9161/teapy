use super::super::{StringExt, TimeExt};
use lazy::Expr;
use tea_core::prelude::*;

#[ext_trait]
impl<'a> ExprTimeExt for Expr<'a> {
    fn strptime(&mut self, fmt: Option<String>) -> &mut Self {
        self.cast_string().chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            let out: ArrOk<'a> = match_arrok!(arr; String(a) => {
                let out = a.view().strptime(fmt.as_deref());
                Ok(out.into())
            },)
            .unwrap();
            Ok((out.into(), ctx))
        });
        self
    }

    fn strftime(&mut self, fmt: Option<String>) -> &mut Self {
        self.cast_datetime(None).chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            let out: ArrOk<'a> = match_arrok!(
                arr;
                Time(a) =>
                {
                    let out = a.view().strftime(fmt.as_deref());
                    Ok(out.into())
                },
            )
            .unwrap();
            Ok((out.into(), ctx))
        });
        self
    }
}
