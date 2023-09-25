use std::sync::Arc;

use super::export::*;
use crate::{lazy::DataDict, Arr1, CollectTrustedToVec, CorrMethod, Data, DateTime, TimeDelta};
use ahash::HashMap;
use ndarray::s;

macro_rules! impl_group_by_time_info_agg {
    ($func_name: ident, $func_1d: ident ($($p: ident: $ty: ty),*) $(.$func2: ident ($($p2: ident: $ty2: ty),*))*) => {
        impl<'a> Expr<'a> {
            pub fn $func_name(&mut self, group_info: Expr<'a>  $(,$p:$ty)* $($(,$p2:$ty2)*)*) -> &mut Self
            {
                self.chain_f_ctx(
                    move |(data, ctx)| {
                        let arr = data.view_arr(ctx.as_ref())?.deref();
                        let group_start = if let Ok(mut group_info) = group_info.view_arr_vec(ctx.as_ref()) {
                            group_info.pop().unwrap().deref().cast_usize()
                        } else {
                            group_info.view_arr(ctx.as_ref())?.deref().cast_usize()
                        };
                        let group_start_view = group_start.view().to_dim1()?;
                        // let len = group_start_view.len();
                        let out: ArrOk<'a> = match_arrok!(numeric arr, arr, {
                            let arr = arr.view().to_dim1()?;
                            let out = group_start_view.as_slice().unwrap().windows(2)
                            .map(|v| {
                                let (start, next_start) = (v[0], v[1]);
                                let current_arr = arr.slice(s![start..next_start]).wrap();
                                current_arr.$func_1d($($p),*)$(.$func2($($p2),*))*
                            })
                            .collect_trusted();
                            Arr1::from_vec(out).to_dimd().into()
                        });
                        Ok((out.into(), ctx.clone()))
                    }
                );
                self
            }
        }
    };

    (in2 $func_name: ident, $func_1d: ident ($($p: ident: $ty: ty),*)) => {
        impl<'a> Expr<'a> {
            pub fn $func_name(&mut self, other: Expr<'a>, group_info: Expr<'a>  $(,$p:$ty)*) -> &mut Self
            {
                self.chain_f_ctx(
                    move |(data, ctx)| {
                        let arr = data.view_arr(ctx.as_ref())?.deref();
                        let other = other.view_arr(ctx.as_ref())?.deref();
                        let group_start = if let Ok(mut group_info) = group_info.view_arr_vec(ctx.as_ref()) {
                            group_info.pop().unwrap().deref().cast_usize()
                        } else {
                            group_info.view_arr(ctx.as_ref())?.deref().cast_usize()
                        };
                        let group_start_view = group_start.view().to_dim1()?;
                        let out: ArrOk<'a> = match_arrok!(numeric arr, arr, {
                            let arr = arr.view().to_dim1()?;
                            let out = group_start_view.as_slice().unwrap().windows(2)
                            .map(|v| {
                                let (start, next_start) = (v[0], v[1]);
                                let current_arr = arr.slice(s![start..next_start]).wrap();
                                match_arrok!(numeric &other, other, {
                                    let other = other.view().to_dim1().unwrap();
                                    debug_assert_eq!(current_arr.len(), other.len());
                                    let other_arr = other.slice(s![start..next_start]).wrap();
                                    current_arr.$func_1d(&other_arr, $($p),*)
                                })

                            })
                            .collect_trusted();
                            Arr1::from_vec(out).to_dimd().into()
                        });
                        Ok((out.into(), ctx.clone()))
                    }
                );
                self
            }
        }
    };
}

impl_group_by_time_info_agg!(group_by_time_max, max_1d());
impl_group_by_time_info_agg!(group_by_time_min, min_1d());
impl_group_by_time_info_agg!(group_by_time_mean, mean_1d(stable: bool));
impl_group_by_time_info_agg!(group_by_time_sum, sum_1d(stable: bool));
impl_group_by_time_info_agg!(group_by_time_std, std_1d(stable: bool));
impl_group_by_time_info_agg!(group_by_time_var, var_1d(stable: bool));
impl_group_by_time_info_agg!(group_by_time_first, first_unwrap());
impl_group_by_time_info_agg!(group_by_time_last, last_unwrap());
impl_group_by_time_info_agg!(group_by_time_valid_first, valid_first_1d());
impl_group_by_time_info_agg!(group_by_time_valid_last, valid_last_1d());

impl_group_by_time_info_agg!(in2 group_by_time_corr, corr_1d(method: CorrMethod, stable: bool));

impl<'a> Expr<'a> {
    /// This func should return an array indicates the start of each group
    /// and a label array indicates the label of each group
    pub fn get_group_by_time_info<TD>(&mut self, duration: TD, closed: String) -> &mut Self
    where
        TD: Into<TimeDelta>,
    {
        let duration: TimeDelta = duration.into();
        self.chain_f_ctx(move |(data, ctx)| {
            let closed = closed.clone();
            let arr = data.view_arr(ctx.as_ref())?.deref().cast_datetime_default();
            let ts = arr.view().to_dim1()?;
            if ts.is_empty() {
                let label = Arr1::from_vec(Vec::<DateTime>::with_capacity(0))
                    .to_dimd()
                    .into();
                let start_vec = Arr1::from_vec(Vec::<usize>::with_capacity(0))
                    .to_dimd()
                    .into();
                return Ok((Data::ArrVec(vec![label, start_vec]), ctx));
            }

            let mut label = vec![];
            let mut start_vec = vec![];
            match closed.to_lowercase().as_str() {
                "left" => {
                    let mut start = ts.first().unwrap().duration_trunc(duration.clone());
                    label.push(start);
                    start_vec.push(0);
                    for i in 0..ts.len() {
                        let t = unsafe { *ts.uget(i) };
                        if t < start + duration.clone() {
                            continue;
                        } else {
                            start_vec.push(i);
                            start = t.duration_trunc(duration.clone());
                            label.push(start);
                        }
                    }
                }
                "right" => {
                    let mut start = ts.first().unwrap().duration_trunc(duration.clone());
                    if start == *ts.get(0).unwrap() {
                        start = start - duration.clone();
                    }
                    label.push(start);
                    start_vec.push(0);
                    for i in 0..ts.len() {
                        let t = unsafe { *ts.uget(i) };
                        if (t <= start + duration.clone()) | (t == start) {
                            continue;
                        } else {
                            start_vec.push(i);
                            start = t.duration_trunc(duration.clone());
                            if start == t {
                                start = start - duration.clone();
                            }
                            label.push(start);
                        }
                    }
                }
                _ => unimplemented!(),
            }
            start_vec.push(ts.len()); // the end of the array, this element is not the start of the group
            let label: ArrOk<'a> = Arr1::from_vec(label).to_dimd().into();
            let start_vec: ArrOk<'a> = Arr1::from_vec(start_vec).to_dimd().into();
            Ok((Data::ArrVec(vec![label, start_vec]), ctx))
        });
        self
    }

    #[allow(unreachable_patterns)]
    pub fn group_by_time(
        &mut self,
        agg_expr: Expr<'a>,
        group_info: Expr<'a>,
        others: Vec<Expr<'a>>,
    ) -> &mut Self {
        let name = self.name().unwrap();
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?.deref();
            let others_ref = others
                .iter()
                .map(|e| e.view_arr(ctx.as_ref()).unwrap().deref())
                .collect_trusted();
            let others_name = others.iter().map(|e| e.name().unwrap()).collect_trusted();
            let group_start = if let Ok(mut group_info) = group_info.view_arr_vec(ctx.as_ref()) {
                group_info.pop().unwrap().deref().cast_usize()
            } else {
                group_info.view_arr(ctx.as_ref())?.deref().cast_usize()
            };
            let group_start_view = group_start.view().to_dim1()?;
            let columns = std::iter::once(name.clone())
                .chain(others_name)
                .collect::<Vec<_>>();
            let mut map: Option<Arc<HashMap<String, usize>>> = None;
            let agg_expr = agg_expr.flatten();
            let out: ArrOk<'a> = match_arrok!(arr, arr, {
                let arr = arr.view().to_dim1()?;
                let out = group_start_view
                    .as_slice()
                    .unwrap()
                    .windows(2)
                    .map(|v| {
                        let (start, next_start) = (v[0], v[1]);
                        let slice = s![start..next_start];
                        let current_arr = arr.slice(slice).wrap();
                        let current_others: Vec<ArrOk> = others_ref
                            .iter()
                            .map(|arr| arr.slice(slice.clone()))
                            .collect_trusted();
                        let exprs: Vec<Expr<'_>> = std::iter::once(current_arr.to_dimd().into())
                            .chain(current_others.into_iter().map(|a| a.into()))
                            .collect::<Vec<_>>();
                        let current_ctx = if map.is_some() {
                            DataDict {
                                data: exprs,
                                map: map.clone().unwrap(),
                            }
                        } else {
                            let dd = DataDict::new(exprs, Some(columns.clone()));
                            map = Some(dd.map.clone());
                            dd
                        };
                        let out_e = agg_expr.context_clone();
                        // this is safe as we don't return a view on the current context
                        // into_owned is important here to guarantee the above
                        let current_ctx: Arc<DataDict> =
                            Arc::new(unsafe { std::mem::transmute(current_ctx) });
                        let o = out_e
                            .view_arr(Some(&current_ctx))
                            .unwrap()
                            .deref()
                            .into_owned();
                        o
                    })
                    .collect_trusted();
                ArrOk::same_dtype_concat_1d(out)
            });
            Ok((out.into(), ctx.clone()))
        });
        self
    }
}
