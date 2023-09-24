use super::export::*;
use crate::{
    ArbArray, Arr, Arr1, Arr2, ArrD, CollectTrustedToVec, CorrMethod, DataType, FillMethod,
};
use ndarray::{Array1, Axis};
use rayon::prelude::*;
#[cfg(feature = "stat")]
use statrs::distribution::ContinuousCDF;
use std::cmp::Ordering;

#[derive(Clone)]
pub enum DropNaMethod {
    Any,
    All,
}

impl<'a> Expr<'a> {
    #[allow(unreachable_patterns)]
    pub fn full(shape: &Expr<'a>, value: Expr<'a>) -> Expr<'a> {
        let mut e = shape.clone();
        e.chain_f_ctx(move |(data, ctx)| {
            // let value = value.clone();
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

    pub fn len(&mut self) -> &mut Self {
        self.chain_f_ctx(|(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            Ok((arr.len().into(), ctx))
        });
        self
    }

    pub fn count_value(&mut self, value: Expr<'a>, axis: i32, par: bool) -> &mut Self {
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

    #[allow(unreachable_patterns)]
    pub fn deep_copy(&mut self) -> &mut Self {
        self.chain_f_ctx(|(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            let out: ArrOk<'a> = match_arrok!(arr, a, { a.view().to_owned().into() });
            Ok((out.into(), ctx))
        });
        self
    }

    pub fn fillna(
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
                a.viewmut().fillna_inplace(method, value, axis, par);
            });
            Ok((arr.into(), ctx))
        });
        self
    }

    pub fn clip(&mut self, min: Expr<'a>, max: Expr<'a>, axis: i32, par: bool) -> &mut Self {
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
                a.viewmut().clip_inplace(min, max, axis, par);
            });
            Ok((arr.into(), ctx))
        });
        self
    }

    pub fn shift(
        &mut self,
        n: Expr<'a>,
        fill: Option<Expr<'a>>,
        axis: i32,
        par: bool,
    ) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            let n = n.view_arr(ctx.as_ref())?.deref().cast_i32().into_owned().into_scalar()?;
            let fill = fill.as_ref().map(|f| f.view_arr(ctx.as_ref()).unwrap().deref());
            use ArrOk::*;
            let out: ArrOk<'a> = if matches!(&arr, I32(_) | I64(_)) {
                arr.deref().cast_f64().view().shift(n, fill.map(|f| f.cast_f64().into_owned().into_scalar().unwrap()), axis, par).into()
            } else {
                match_arrok!(castable arr, a, {
                    let f = fill.map(|f| match_arrok!(castable f, f, {f.into_owned().into_scalar().unwrap().cast()}));
                    a.view().shift(n, f, axis, par).into()
                })
            };
            Ok((out.into(), ctx))
        });
        self
    }

    pub fn any(&mut self, axis: i32, par: bool) -> &mut Self {
        self.cast_bool().chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            Ok((
                match_arrok!(arr, arr, { arr.view().any(axis, par).into() }, Bool),
                ctx,
            ))
        });
        self
    }

    pub fn all(&mut self, axis: i32, par: bool) -> &mut Self {
        self.cast_bool().chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            Ok((
                match_arrok!(arr, arr, { arr.view().all(axis, par).into() }, Bool),
                ctx,
            ))
        });
        self
    }

    pub fn abs(&mut self) -> &mut Self {
        self.chain_f_ctx(|(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            let out: ArrOk<'a> =
                match_arrok!(arr, a, { a.view().abs().into() }, F32, F64, I32, I64);
            Ok((out.into(), ctx))
        });
        self
    }

    pub fn sign(&mut self) -> &mut Self {
        self.chain_f_ctx(|(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            let out: ArrOk<'a> =
                match_arrok!(arr, a, { a.view().sign().into() }, F32, F64, I32, I64);
            Ok((out.into(), ctx))
        });
        self
    }

    pub fn round(&mut self, precision: u32) -> &mut Self {
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

    pub fn pow(&mut self, n: Expr<'a>, par: bool) -> &mut Self {
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

    #[cfg(feature = "stat")]
    pub fn t_cdf(&mut self, df: Expr<'a>, loc: Option<f64>, scale: Option<f64>) -> &mut Self {
        use statrs::distribution::StudentsT;
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?.cast_f64();
            let df = df
                .view_arr(ctx.as_ref())?
                .deref()
                .cast_f64()
                .into_owned()
                .into_scalar()?;
            let loc = loc.unwrap_or(0.);
            let scale = scale.unwrap_or(1.);
            let n = StudentsT::new(loc, scale, df).unwrap();
            let out = arr.view().map(|v| n.cdf(*v));
            Ok((out.into(), ctx))
        });
        self
    }

    #[cfg(feature = "stat")]
    pub fn norm_cdf(&mut self, mean: Option<f64>, std: Option<f64>) -> &mut Self {
        use statrs::distribution::Normal;
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?.cast_f64();
            let n = Normal::new(mean.unwrap_or(0.), std.unwrap_or(1.)).unwrap();
            let out = arr.view().map(|v| n.cdf(*v));
            Ok((out.into(), ctx))
        });
        self
    }

    #[cfg(feature = "stat")]
    pub fn f_cdf(&mut self, df1: Expr<'a>, df2: Expr<'a>) -> &mut Self {
        use statrs::distribution::FisherSnedecor;
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?.cast_f64();
            let df1 = df1
                .view_arr(ctx.as_ref())?
                .deref()
                .cast_f64()
                .into_owned()
                .into_scalar()?;
            let df2 = df2
                .view_arr(ctx.as_ref())?
                .deref()
                .cast_f64()
                .into_owned()
                .into_scalar()?;
            let n = FisherSnedecor::new(df1, df2).unwrap();
            let out = arr.view().map(|v| n.cdf(*v));
            Ok((out.into(), ctx))
        });
        self
    }

    #[allow(unreachable_patterns)]
    pub fn filter(&mut self, mask: Expr<'a>, axis: Expr<'a>, par: bool) -> &mut Self {
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

    pub fn dropna(&mut self, axis: Expr<'a>, how: DropNaMethod, par: bool) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            let ndim = arr.ndim();
            let axis = axis.view_arr(ctx.as_ref())?.deref().cast_i32().into_owned().into_scalar()?;
            let out: ArrOk<'a> = match_arrok!(numeric arr, a, {
                match ndim {
                    1 => a.view().to_dim1()?.remove_nan_1d().to_dimd().into(),
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
                        a.filter(&mask, axis, par).to_dimd().into()
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
    pub fn select(&mut self, slc: Expr<'a>, axis: Expr<'a>, check: bool) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let slc_e = slc.clone();
            let mut slc = slc_e.view_arr(ctx.as_ref())?.deref();
            let arr = data.view_arr(ctx.as_ref())?;
            // let mut slc = slc.into_arr(ctx.clone())?;
            if slc.ndim() > 1 {
                return Err("The slice must be dim 0 or dim 1 when select on axis".into());
            }
            let axis_i32 = axis
                .view_arr(ctx.as_ref())?
                .deref()
                .cast_i32()
                .into_owned()
                .into_scalar()?;
            let out: ArrOk<'a> = match_arrok!(arr, a, {
                let a_view = a.view();
                let axis = a_view.norm_axis(axis_i32);
                let length = a_view.len_of(axis);
                if matches!(&slc, ArrOk::OptUsize(_)) {
                    // take opiton_usize
                    let slc = slc.deref().cast_optusize();
                    let slc_view = slc.view();
                    match a_view.dtype() {
                        DataType::I32 => {
                            if slc_view.len() == 1 {
                                unsafe { a_view.into_dtype::<i32>() }
                                    .cast::<f64>()
                                    .index_axis(axis, slc_view.to_dim1()?[0].unwrap())
                                    .to_owned()
                                    .wrap()
                                    .into()
                            } else {
                                unsafe {
                                    a_view
                                        .into_dtype::<i32>()
                                        .cast::<f64>()
                                        .take_option_clone_unchecked(
                                            slc_view.to_dim1()?,
                                            axis.index() as i32,
                                            false,
                                        )
                                }
                                .into()
                            }
                        }
                        DataType::I64 => {
                            if slc_view.len() == 1 {
                                unsafe { a_view.into_dtype::<i64>() }
                                    .cast::<f64>()
                                    .index_axis(axis, slc_view.to_dim1()?[0].unwrap())
                                    .to_owned()
                                    .wrap()
                                    .into()
                            } else {
                                unsafe {
                                    a_view
                                        .into_dtype::<i64>()
                                        .cast::<f64>()
                                        .take_option_clone_unchecked(
                                            slc_view.to_dim1()?,
                                            axis.index() as i32,
                                            false,
                                        )
                                }
                                .into()
                            }
                        }
                        _ => {
                            if slc_view.len() == 1 {
                                a_view
                                    .index_axis(axis, slc_view.to_dim1()?[0].unwrap())
                                    .to_owned()
                                    .wrap()
                                    .into()
                            } else {
                                unsafe {
                                    a_view.take_option_clone_unchecked(
                                        slc_view.to_dim1()?,
                                        axis.index() as i32,
                                        false,
                                    )
                                }
                                .into()
                            }
                        }
                    }
                } else if matches!(&slc, ArrOk::Bool(_)) {
                    let slc = slc.deref().cast_bool();
                    let slc_view = slc.view();
                    a_view.filter(&slc_view.to_dim1()?, axis_i32, false).into()
                } else {
                    if check {
                        slc = match slc {
                            ArrOk::I32(slc) => slc
                                .view()
                                .to_dim1()?
                                .map(|s| a_view.ensure_index(*s, length))
                                .to_dimd()
                                .into(),
                            ArrOk::I64(slc) => slc
                                .view()
                                .to_dim1()?
                                .map(|s| a_view.ensure_index(*s as i32, length))
                                .to_dimd()
                                .into(),
                            _ => slc,
                        };
                    }
                    let slc = slc.cast_usize();
                    let slc_view = slc.view();
                    if slc_view.len() == 1 {
                        a_view
                            .index_axis(axis, slc_view.to_dim1()?[0])
                            .to_owned()
                            .wrap()
                            .into()
                    } else {
                        if check {
                            a_view
                                .select(axis, &slc_view.to_dim1()?.as_slice().unwrap())
                                .wrap()
                                .into()
                        } else {
                            a_view
                                .select_unchecked(axis, &slc_view.to_dim1()?.as_slice().unwrap())
                                .into()
                        }
                    }
                }
            });
            Ok((out.into(), ctx))
        });
        self
    }

    pub fn where_(&mut self, mask: Expr<'a>, value: Expr<'a>, par: bool) -> &mut Self {
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

    #[allow(unreachable_patterns)]
    pub fn shape(&mut self) -> &mut Self {
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

    pub fn concat(&mut self, other: Vec<Expr<'a>>, axis: i32) -> &mut Self {
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

    pub fn stack(&mut self, other: Vec<Expr<'a>>, axis: i32) -> &mut Self {
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

    pub fn get_sort_idx(&mut self, by: Vec<Expr<'a>>, rev: bool) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            if arr.ndim() != 1 {
                return Err("Currently only 1 dim Expr can be sorted".into());
            }
            let by_e = by.to_vec();
            let by = by_e
                .par_iter()
                .map(|e| e.view_arr(ctx.as_ref()).unwrap().deref())
                .collect::<Vec<_>>();
            let len = arr.len();
            let mut idx = Vec::from_iter(0..len);
            use ArrOk::*;
            idx.sort_by(move |a, b| {
                let mut order = Ordering::Equal;
                for arr in by.iter() {
                    let rtn = match &arr {
                        String(_) | DateTime(_) | TimeDelta(_) => {
                            match_arrok!(
                                arr,
                                arr,
                                {
                                    let key_view = arr
                                        .view()
                                        .to_dim1()
                                        .expect("Currently only 1 dim array can be sort key");
                                    let (va, vb) =
                                        unsafe { (key_view.uget(*a), key_view.uget(*b)) };
                                    if !rev {
                                        va.cmp(vb)
                                    } else {
                                        va.cmp(vb).reverse()
                                    }
                                },
                                String,
                                DateTime
                            )
                        }
                        _ => {
                            match_arrok!(
                                numeric arr,
                                arr,
                                {
                                    let key_view = arr.view().to_dim1().expect(
                                        "Currently only 1 dim array can be sort key",
                                    );
                                    let (va, vb) =
                                        unsafe { (key_view.uget(*a), key_view.uget(*b)) };
                                    if !rev {
                                        va.nan_sort_cmp_stable(vb)
                                    } else {
                                        va.nan_sort_cmp_rev_stable(vb)
                                    }
                                }
                            )
                        }
                    };
                    if rtn != Ordering::Equal {
                        order = rtn;
                        break;
                    }
                }
                order
            });
            Ok((Arr1::from_vec(idx).to_dimd().into(), ctx))
        });
        self
    }

    pub fn sort(&mut self, by: Vec<Expr<'a>>, rev: bool) -> &mut Self {
        let mut idx = self.clone();
        idx.get_sort_idx(by, rev);
        self.select(idx, 0.into(), false);
        self
    }

    #[allow(unreachable_patterns)]
    pub fn split_vec_base(self, len: usize) -> Vec<Self> {
        // todo: improve performance
        let mut out = (0..len).map(|_| self.clone()).collect_trusted();
        out.iter_mut().enumerate().for_each(|(i, e)| {
            e.chain_f_ctx(move |(data, ctx)| {
                let arr = data.view_arr_vec(ctx.as_ref())?.remove(i);
                Ok((match_arrok!(arr, a, { a.view().to_owned().into() }), ctx))
            });
        });
        out
    }

    pub fn strptime(&mut self, fmt: String) -> &mut Self {
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

    pub fn strftime(&mut self, fmt: Option<String>) -> &mut Self {
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

pub fn corr<'a>(exprs: Vec<Expr<'a>>, method: CorrMethod, stable: bool) -> Expr<'a> {
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
                            arri.deref().view().to_dim1()?.corr_1d(&arrj.deref().view().to_dim1()?, method, stable)
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
