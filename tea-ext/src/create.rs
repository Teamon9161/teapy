use lazy::Expr;
use ndarray::Array1;
use tea_core::prelude::*;

#[ext_trait]
impl<'a> CreateExt for Expr<'a> {
    #[allow(unreachable_patterns)]
    pub fn full(shape: &Expr<'a>, value: Expr<'a>) -> Expr<'a> {
        let mut e = shape.clone();
        e.chain_f_ctx(move |(data, ctx)| {
            let value = value.clone().into_arr(ctx.clone())?;
            let shape = data.into_arr(ctx.clone())?.cast_usize();
            let ndim = shape.ndim();
            if ndim == 0 {
                let shape = shape.into_owned().into_scalar()?;
                match_arrok!(value, v, {
                    let v = v.into_owned().into_scalar()?;
                    Ok((Arr::from_elem(shape, v).to_dimd().into(), ctx))
                })
            } else if ndim == 1 {
                let shape = shape.view().to_dim1()?;
                match_arrok!(value, v, {
                    let v = v.into_owned().into_scalar()?;
                    Ok((
                        Arr::from_elem(shape.to_slice().unwrap(), v)
                            .to_dimd()
                            .into(),
                        ctx,
                    ))
                })
            } else {
                Err("The dim of shape should not be greater than 1".into())
            }
        });
        e
    }

    pub fn arange(start: Option<Expr<'a>>, end: &Expr<'a>, step: Option<Expr<'a>>) -> Expr<'a> {
        let mut e = end.clone();
        e.chain_f_ctx(move |(data, ctx)| {
            // let start_e = start.clone();
            let start = start.as_ref().map(|s| s.view_arr(ctx.as_ref()).unwrap());
            let end = data.into_arr(ctx.clone())?;
            let step = step.as_ref().map(|s| s.view_arr(ctx.as_ref()).unwrap());

            let start = start
                .map(|s| s.deref().cast_f64().into_owned().into_scalar().unwrap())
                .unwrap_or(0.);
            let end = end.cast_f64().into_owned().into_scalar()?;
            let step = step
                .map(|s| s.deref().cast_f64().into_owned().into_scalar().unwrap())
                .unwrap_or(1.);
            Ok((Array1::range(start, end, step).wrap().to_dimd().into(), ctx))
        });
        e
    }
}
