use super::super::{StringExt, TimeExt};
use lazy::Expr;
use tea_core::prelude::*;

#[ext_trait]
impl<'a> ExprTimeExt for Expr<'a> {
    fn strptime(&mut self, fmt: String) -> &mut Self {
        self.cast_string().chain_f_ctx(move |(data, ctx)| {
            let fmt = fmt.clone();
            let arr = data.view_arr(ctx.as_ref())?;
            let out: ArrOk<'a> = match_arrok!(
                arr,
                a,
                {
                    let out = a.view().strptime(fmt);
                    out.into()
                },
                String
            );
            Ok((out.into(), ctx))
        });
        self
    }

    fn strftime(&mut self, fmt: Option<String>) -> &mut Self {
        self.cast_datetime_default()
            .chain_f_ctx(move |(data, ctx)| {
                let arr = data.view_arr(ctx.as_ref())?;
                let out: ArrOk<'a> = match_arrok!(
                    arr,
                    a,
                    {
                        let out = a.view().strftime(fmt.as_deref());
                        out.into()
                    },
                    DateTime
                );
                Ok((out.into(), ctx))
            });
        self
    }
}
