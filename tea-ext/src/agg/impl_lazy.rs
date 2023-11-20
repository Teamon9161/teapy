use core::prelude::*;
// use ndarray::{Data, Dimension, Zip};
use super::*;
use rayon::prelude::*;
// use crate::auto_impl_view;

#[cfg(feature = "lazy")]
use lazy::Expr;

#[cfg(feature = "lazy")]
macro_rules! auto_impl_agg_view {
    (
        $(in1, [$($func: ident),* $(,)?], $other: tt);*
        $(;in2, [$($func2: ident),* $(,)?], $other2: tt)*
        $(;)?

    ) => {
        #[ext_trait]
        impl<'a> AutoExprAggExt for Expr<'a> {
            $($(auto_impl_view!(in1, $func, $other);)*)*
            $($(auto_impl_view!(in2, $func2, $other2);)*)*
        }
    };
}

#[cfg(feature = "lazy")]
auto_impl_agg_view!(
    in1, [
        argmax, argmin, count_nan, count_notnan, median,
        max, min, prod, first, last, valid_first, valid_last
    ],
    (axis: i32, par: bool);
    in1, [ndim], ();
    in1, [sum], (stable: bool, axis: i32, par: bool);
    in1, [mean, var, std, skew, kurt], (min_periods: usize, stable: bool, axis: i32, par: bool);
    in1, [quantile], (q: f64, method: QuantileMethod, axis: i32, par: bool);
    in2, [corr], (method: CorrMethod, min_periods: usize, stable: bool, axis: i32, par: bool);
    in2, [cov], (min_periods: usize, stable: bool, axis: i32, par: bool);
);

#[ext_trait]
impl<'a> ExprAggExt for Expr<'a> {
    fn len(&mut self) -> &mut Self {
        self.chain_f_ctx(|(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            Ok((arr.len().into(), ctx))
        });
        self
    }

    fn count_value(&mut self, value: Expr<'a>, axis: i32, par: bool) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            let value = value.view_arr(ctx.as_ref())?;
            let out = match_arrok!(castable arr, a, {
                let value = match_arrok!(castable value, v, {v.deref().into_owned().into_scalar()?.cast()});
                a.view().count_v(value, axis, par)
            });
            Ok((out.into(), ctx))
        });
        self
    }

    fn any(&mut self, axis: i32, par: bool) -> &mut Self {
        self.cast_bool().chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            Ok((
                match_arrok!(arr, arr, { arr.view().any(axis, par).into() }, Bool),
                ctx,
            ))
        });
        self
    }

    fn all(&mut self, axis: i32, par: bool) -> &mut Self {
        self.cast_bool().chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            Ok((
                match_arrok!(arr, arr, { arr.view().all(axis, par).into() }, Bool),
                ctx,
            ))
        });
        self
    }

    #[allow(unreachable_patterns)]
    fn shape(&mut self) -> &mut Self {
        self.chain_f_ctx(|(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            let out: ArrOk<'a> = match_arrok!(arr, a, {
                let shape = a.view().shape().to_owned();
                Arr1::from_vec(shape).to_dimd().into()
            });
            Ok((out.into(), ctx))
        });
        self
    }
}

pub fn corr<'a>(
    exprs: Vec<Expr<'a>>,
    method: CorrMethod,
    min_periods: usize,
    stable: bool,
) -> Expr<'a> {
    let mut out: Expr<'a> = Default::default();
    out.chain_f_ctx(move |(_, ctx)| {
        // let exprs = exprs.iter().map(|v| (*v).clone()).collect_trusted();
        let all_arr = exprs.par_iter().map(|e| e.view_arr(ctx.as_ref()).unwrap()).collect::<Vec<_>>();
        let len = all_arr.len();
        let mut corr_arr = Arr2::<f64>::uninit((len, len));
        for i in 0..len {
            for j in i..len {
                let corr = if i != j {
                    let arri = *unsafe{all_arr.get_unchecked(i)};
                    let arrj = *unsafe{all_arr.get_unchecked(j)};
                    match_arrok!(numeric arri, arri, {
                        match_arrok!(numeric arrj, arrj, {
                            arri.deref().view().to_dim1()?.corr_1d(&arrj.deref().view().to_dim1()?, method, min_periods, stable)
                        })
                    })
                } else {
                    1.0
                };
                unsafe {
                    corr_arr.uget_mut((i, j)).write(corr);
                    corr_arr.uget_mut((j, i)).write(corr);
                }
            }
        }
        let corr_arr: ArrD<f64> = unsafe { corr_arr.assume_init()}.to_dimd();
        Ok((corr_arr.into(), ctx))
    });
    out
}
