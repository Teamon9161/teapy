use super::*;
use lazy::ColumnSelector;
use rayon::prelude::*;
use tea_core::prelude::*;

#[cfg(feature = "lazy")]
use lazy::{DataDict, Expr};

#[ext_trait(lazy_only, lazy = "view")]
impl<'a> AggExt for Expr<'a> {
    fn len(&self) {}

    fn ndim(&self) {}

    #[teapy(type = "numeric")]
    fn sum(&self, stable: bool, axis: i32, par: bool) {}

    #[teapy(type = "numeric")]
    fn mean(&self, min_periods: usize, stable: bool, axis: i32, par: bool) {}

    #[teapy(type = "numeric")]
    fn min(&self, axis: i32, par: bool) {}

    #[teapy(type = "numeric")]
    fn max(&self, axis: i32, par: bool) {}
}

#[ext_trait]
impl<'a> ExprAggExt for Expr<'a> {
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

#[ext_trait]
impl<'a> DataDictCorrExt for DataDict<'a> {
    pub fn corr<'b>(
        &'b self,
        col: Option<ColumnSelector<'b>>,
        method: CorrMethod,
        min_periods: usize,
        stable: bool,
    ) -> Expr<'a> {
        let col: ColumnSelector<'_> = col.unwrap_or(ColumnSelector::All);
        let exprs: Vec<&Expr<'a>> = self.get(col).unwrap().into_exprs();
        let exprs: Vec<Expr<'a>> = exprs.into_iter().cloned().collect::<Vec<_>>();
        corr(exprs, method, min_periods, stable)
    }
}
