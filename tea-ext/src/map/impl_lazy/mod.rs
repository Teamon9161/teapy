// mod auto_impl;
mod impl_view;
#[cfg(feature = "stat")]
mod stat;
#[cfg(feature = "time")]
mod time;

// pub use auto_impl::{AutoExprInplaceExt, AutoExprMapExt};
pub use impl_view::ExprViewExt;
#[cfg(feature = "stat")]
pub use stat::ExprStatExt;
#[cfg(feature = "time")]
pub use time::ExprTimeExt;

use super::super::*;
use lazy::{adjust_slice, Expr};
use ndarray::{Axis, SliceInfoElem};
use rayon::prelude::*;
use tea_core::prelude::*;
// use tea_core::utils::CollectTrustedToVec; // use map trait of ArrBase

#[ext_trait]
impl<'a> ExprInplaceExt for Expr<'a> {
    #[allow(unreachable_patterns)]
    pub fn put_mask(&mut self, mask: Expr<'a>, value: Expr<'a>, axis: i32, par: bool) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let mut mask = mask.clone();
            let mut value = value.clone();
            mask.cast_bool().eval_inplace(ctx.clone())?;
            value.eval_inplace(ctx.clone())?;
            let mask_arr = mask.view_arr(ctx.as_ref())?;
            let mask_arr = match_arrok!(mask_arr, a, { a }, Bool);
            let value_arr = value.into_arr(ctx.clone())?;
            let mut arr = data.into_arr(ctx.clone())?;
            match_arrok!(&mut arr, a, {
                let value: ArbArray<_> = value_arr.cast();
                a.view_mut()
                    .put_mask(&mask_arr.view(), &value.view(), axis, par)?
            });
            Ok((arr.into(), ctx))
        });
        self
    }

    fn shift(&mut self, n: Expr<'a>, fill: Option<Expr<'a>>, axis: i32, par: bool) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let mut arr = data.into_arr(ctx.clone())?;
            let n = n.view_arr(ctx.as_ref())?.deref().cast_i32().into_owned().into_scalar()?;
            let fill = fill.as_ref().map(|f| f.view_arr(ctx.as_ref()).unwrap().deref());
            use ArrOk::*;
            let arr = if matches!(&arr, I32(_) | I64(_)) {
                let mut arr = arr.cast_f64();
                arr.view_mut().shift(n,fill.map(|f| f.cast_f64().into_owned().into_scalar().unwrap()), axis, par);
                arr.into()
            } else {
                match_arrok!(castable &mut arr, a, {
                    let f = fill.map(|f| match_arrok!(castable f, f, {f.into_owned().into_scalar().unwrap().cast()}));
                    a.view_mut().shift(n, f, axis, par);
                });
                arr
            };
            Ok((arr.into(), ctx))
        });
        self
    }

    fn diff(&mut self, n: Expr<'a>, fill: Option<Expr<'a>>, axis: i32, par: bool) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let mut arr = data.into_arr(ctx.clone())?;
            let n = n.view_arr(ctx.as_ref())?.deref().cast_i32().into_owned().into_scalar()?;
            let fill = fill.as_ref().map(|f| f.view_arr(ctx.as_ref()).unwrap().deref());
            use ArrOk::*;
            let out: ArrOk<'a> = if matches!(&arr, I32(_) | I64(_)) {
                let mut arr = arr.cast_f64();
                arr.view_mut().diff(n, fill.map(|f| f.cast_f64().into_owned().into_scalar().unwrap()), axis, par);
                arr.into()
            } else {
                match_arrok!(castable &mut arr, a, {
                    let f = fill.map(|f| match_arrok!(castable f, f, {f.into_owned().into_scalar().unwrap().cast()}));
                    a.view_mut().shift(n, f, axis, par)
                });
                arr
            };
            Ok((out.into(), ctx))
        });
        self
    }

    pub fn set_item_by_slice(&mut self, slc: Vec<SliceInfoElem>, value: Self) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let value = value.view_arr(ctx.as_ref())?;
            let mut arr = data.into_arr(ctx.clone())?;
            let slc_info = adjust_slice(slc.clone(), arr.shape(), arr.ndim());
            match_arrok!(castable &mut arr, arr, {
                let mut arr_view_mut = arr.view_mut();
                let mut arr_mut = arr_view_mut.slice_mut(slc_info).wrap().to_dimd();
                match_arrok!(castable value, v, {
                    let v = v.deref().cast();
                    arr_mut.assign(&v.view());
                })
            });
            Ok((arr.into(), ctx))
        });
        self
    }

    fn fillna(
        &mut self,
        method: FillMethod,
        value: Option<Expr<'a>>,
        axis: i32,
        par: bool,
    ) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let mut arr = data.into_arr(ctx.clone())?;
            let value = value.as_ref().map(|v| {
                v.view_arr(ctx.as_ref())
                    .unwrap()
                    .deref()
                    .cast_f64()
                    .into_owned()
                    .into_scalar()
                    .unwrap()
            });
            match_arrok!(numeric &mut arr, a, {
                a.view_mut().fillna(method, value, axis, par);
            });
            Ok((arr.into(), ctx))
        });
        self
    }

    fn clip(&mut self, min: Expr<'a>, max: Expr<'a>, axis: i32, par: bool) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let mut arr = data.into_arr(ctx.clone())?;
            let min = min
                .view_arr(ctx.as_ref())?
                .deref()
                .cast_f64()
                .into_owned()
                .into_scalar()?;
            let max = max
                .view_arr(ctx.as_ref())?
                .deref()
                .cast_f64()
                .into_owned()
                .into_scalar()?;
            match_arrok!(numeric &mut arr, a, {
                a.view_mut().clip(min, max, axis, par);
            });
            Ok((arr.into(), ctx))
        });
        self
    }
}

#[derive(Clone)]
pub enum DropNaMethod {
    Any,
    All,
}

#[ext_trait]
impl<'a> ExprMapExt for Expr<'a> {
    fn is_in(&mut self, other: Expr<'a>) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            let other = other.view_arr(ctx.as_ref())?;
            let out = match_arrok!(castable arr, a, {
                let other: ArbArray<_> = other.deref().cast();
                let other_slc = other.view().to_dim1()?.to_slice().unwrap();
                a.view().is_in(other_slc)
            });
            Ok((out.into(), ctx))
        });
        self
    }

    #[allow(unreachable_patterns)]
    fn deep_copy(&mut self) -> &mut Self {
        self.chain_f_ctx(|(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            let out: ArrOk<'a> = match_arrok!(arr, a, { a.view().to_owned().into() });
            Ok((out.into(), ctx))
        });
        self
    }

    fn abs(&mut self) -> &mut Self {
        self.chain_f_ctx(|(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            let out: ArrOk<'a> =
                match_arrok!(arr, a, { a.view().abs().into() }, F32, F64, I32, I64);
            Ok((out.into(), ctx))
        });
        self
    }

    fn sign(&mut self) -> &mut Self {
        self.chain_f_ctx(|(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            let out: ArrOk<'a> =
                match_arrok!(arr, a, { a.view().sign().into() }, F32, F64, I32, I64);
            Ok((out.into(), ctx))
        });
        self
    }

    fn round(&mut self, precision: u32) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?.cast_f64();
            let out: ArrOk<'a> = arr
                .view()
                .map(|v| {
                    let scale = 10_i32.pow(precision) as f64;
                    (v * scale).round() / scale
                })
                .into();
            Ok((out.into(), ctx))
        });
        self
    }

    fn pow(&mut self, n: Expr<'a>, par: bool) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            let n = n.view_arr(ctx.as_ref())?.deref();
            let out: ArrOk<'a> = if arr.is_int() {
                if n.is_float() {
                    match_arrok!(int arr, a, {
                        a.view().pow(&n.cast_usize().view(), par).into()
                    })
                } else {
                    arr.deref()
                        .cast_f64()
                        .view()
                        .powf(&n.cast_f64().view(), par)
                        .into()
                }
            } else if n.is_float() {
                arr.deref()
                    .cast_f64()
                    .view()
                    .powf(&n.cast_f64().view(), par)
                    .into()
            } else {
                match_arrok!(float arr, a, {
                    a.view().powi(&n.cast_i32().view(), par).into()
                })
            };
            Ok((out.into(), ctx))
        });
        self
    }

    #[allow(unreachable_patterns)]
    fn filter(&mut self, mask: Expr<'a>, axis: Expr<'a>, par: bool) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            let mask = mask.view_arr(ctx.as_ref())?.deref().cast_bool();
            let axis = axis
                .view_arr(ctx.as_ref())?
                .deref()
                .cast_i32()
                .into_owned()
                .into_scalar()?;
            let out: ArrOk<'a> = match_arrok!(arr, a, {
                a.view().filter(&mask.view().to_dim1()?, axis, par).into()
            });
            Ok((out.into(), ctx))
        });
        self
    }

    fn dropna(&mut self, axis: Expr<'a>, how: DropNaMethod, par: bool) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            let ndim = arr.ndim();
            let axis = axis.view_arr(ctx.as_ref())?.deref().cast_i32().into_owned().into_scalar()?;
            let out: ArrOk<'a> = match_arrok!(numeric arr, a, {
                match ndim {
                    1 => a.view().to_dim1()?.dropna_1d().to_dimd().into(),
                    2 => {
                        let a = a.view().to_dim2()?;
                        let axis_n = a.norm_axis(axis);
                        let mask = match (axis_n, how.clone()) {
                            (Axis(0), DropNaMethod::Any) => a.not_nan().all(1, par),
                            (Axis(0), DropNaMethod::All) => a.not_nan().any(1, par),
                            (Axis(1), DropNaMethod::Any) => a.not_nan().all(0, par),
                            (Axis(1), DropNaMethod::All) => a.not_nan().any(0, par),
                            _ => return Err("axis should be 0 or 1 and how should be any or all".into()),
                        };
                        a.filter(&mask.as_dim1(), axis, par).to_dimd().into()
                    },
                    dim => return Err(format!(
                        "dropna only support 1d and 2d array currently, but the array is dim {dim}"
                    )
                    .into()),
                }
            });
            Ok((out.into(), ctx))
        });
        self
    }

    #[allow(unreachable_patterns)]
    fn select(&mut self, slc: Expr<'a>, axis: Expr<'a>, check: bool) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let slc = slc.view_arr(ctx.as_ref())?;
            let arr = data.view_arr(ctx.as_ref())?;
            let axis = axis
                .view_arr(ctx.as_ref())?
                .deref()
                .cast_i32()
                .into_owned()
                .into_scalar()?;
            let out = arr.select(slc, axis, check)?;
            Ok((out.into(), ctx))
        });
        self
    }

    fn where_(&mut self, mask: Expr<'a>, value: Expr<'a>, par: bool) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            let mut mask = mask.clone();
            let value = value.clone();
            mask.cast_bool();
            let mask = mask.view_arr(ctx.as_ref())?;
            let mask_view = match_arrok!(mask, m, { m.view() }, Bool);
            let value = value.view_arr(ctx.as_ref())?.deref();
            let out: ArrOk<'a> = if arr.is_int() & value.is_float() {
                match_arrok!(castable arr.deref().cast_float(), a, {
                    let value: ArbArray<_> = value.cast();
                    a.view().where_(&mask_view, &value.view(), par).into()
                })
            } else {
                match_arrok!(castable arr, a, {
                    let value: ArbArray<_> = value.cast();
                    a.view().where_(&mask_view, &value.view(), par).into()
                })
            };
            Ok((out.into(), ctx))
        });
        self
    }

    #[cfg(feature = "concat")]
    fn concat(&mut self, other: Vec<Expr<'a>>, axis: i32) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            // let other = other.into_par_iter().map(|e| e.view_arr(ctx.as_ref()).unwrap()).collect::<Vec<_>>();
            let other_ref = other
                .par_iter()
                .map(|e| e.view_arr(ctx.as_ref()).unwrap())
                .collect::<Vec<_>>();
            let out: ArrOk<'a> = match_arrok!(castable arr, a, {
                let a = a.deref();
                let a_view = a.view().no_dim0();
                let axis = a_view.norm_axis(axis);
                let other = other_ref.into_par_iter().map(|o| {
                    let o: ArbArray<_> = o.deref().cast();
                    o
                }).collect::<Vec<_>>();
                a.concat(other, axis).into()
            });
            Ok((out.into(), ctx))
        });
        self
    }

    #[cfg(feature = "concat")]
    fn stack(&mut self, other: Vec<Expr<'a>>, axis: i32) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            // let other = other.into_par_iter().map(|e| e.into_arr(ctx.clone()).unwrap()).collect::<Vec<_>>();
            let other_ref = other
                .par_iter()
                .map(|e| e.view_arr(ctx.as_ref()).unwrap())
                .collect::<Vec<_>>();
            let out: ArrOk<'a> = match_arrok!(castable arr, a, {
                let a = a.deref();
                let a_view = a.view().no_dim0();
                let axis = if axis < 0 {
                    Axis(a_view.norm_axis(axis).index() + 1)
                } else {
                    Axis(axis as usize)
                };
                let other = other_ref.into_par_iter().map(|o| {
                    let o: ArbArray<_> = o.deref().cast();
                    o
                }).collect::<Vec<_>>();
                a.stack(other, axis).into()
            });
            Ok((out.into(), ctx))
        });
        self
    }

    fn get_sort_idx(by: Vec<Expr<'a>>, rev: bool) -> Expr<'a> {
        let mut e: Expr<'a> = 0.into();
        e.chain_f_ctx(move |(_data, ctx)| {
            // let arr = data.view_arr(ctx.as_ref())?;
            let by = by
                .par_iter()
                .map(|e| e.view_arr(ctx.as_ref()).unwrap())
                .collect::<Vec<_>>();
            let idx = ArrOk::get_sort_idx(&by, rev)?;
            Ok((Arr1::from_vec(idx).to_dimd().into(), ctx))
        });
        e
    }

    fn sort(&mut self, by: Vec<Expr<'a>>, rev: bool) -> &mut Self {
        // let mut idx = self.clone();
        let idx = Expr::get_sort_idx(by, rev);
        self.select(idx, 0.into(), false);
        self
    }
}
