use std::iter::zip;

use crate::agg::*;
use lazy::Expr;
use tea_core::prelude::*;
use tea_core::utils::CollectTrustedToVec;

use ndarray::{s, Axis};

#[allow(dead_code)]
#[cfg(feature = "time")]
pub enum RollingTimeStartBy {
    Full,
    DurationStart,
}

#[ext_trait]
impl<'a> RollingExt for Expr<'a> {
    // #[cfg(feature = "concat")]
    // #[allow(unreachable_patterns)]
    // pub fn rolling_apply_with_start(
    //     &mut self,
    //     agg_expr: Expr<'a>,
    //     roll_start: Expr<'a>,
    //     others: Vec<Expr<'a>>,
    //     _par: bool,
    // ) -> &mut Self {
    //     use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
    //     use std::sync::Arc;
    //     use tea_hash::TpHashMap;
    //     use tea_lazy::DataDict;
    //     let name = self.name().unwrap();
    //     self.chain_f_ctx(move |(data, ctx)| {
    //         let arr = data.view_arr(ctx.as_ref())?.deref();
    //         let others_ref = others
    //             .par_iter()
    //             .map(|e| e.view_arr(ctx.as_ref()).unwrap().deref())
    //             .collect::<Vec<_>>();
    //         let others_name = others.iter().map(|e| e.name().unwrap()).collect_trusted();
    //         let roll_start = roll_start.view_arr(ctx.as_ref())?.deref().cast_usize();
    //         let roll_start_arr = roll_start.view().to_dim1()?;
    //         let len = arr.len();
    //         if len != roll_start_arr.len() {
    //             return Err(format!(
    //                 "rolling_apply_with_start: arr.len() != roll_start.len(): {} != {}",
    //                 arr.len(),
    //                 roll_start_arr.len()
    //             )
    //             .into());
    //         }
    //         let columns = std::iter::once(name.clone())
    //             .chain(others_name)
    //             .collect::<Vec<_>>();
    //         let mut map: Option<Arc<TpHashMap<String, usize>>> = None;
    //         let agg_expr = agg_expr.flatten();
    //         let out: ArrOk<'a> = match_arrok!(arr, arr, {
    //             let arr = arr.view().to_dim1()?;
    //             let out = zip(roll_start_arr, 0..len)
    //                 .map(|(mut start, end)| {
    //                     if start > end {
    //                         start = end; // the start idx should be inbound
    //                     }
    //                     let current_arr = arr.slice(s![start..end + 1]).wrap();
    //                     let current_others: Vec<ArrOk> = others_ref
    //                         .iter()
    //                         .map(|arr| arr.slice(s![start..=end]))
    //                         .collect_trusted();
    //                     let exprs: Vec<Expr<'_>> = std::iter::once(current_arr.to_dimd().into())
    //                         .chain(current_others.into_iter().map(|a| a.into()))
    //                         .collect::<Vec<_>>();
    //                     let current_ctx = if map.is_some() {
    //                         DataDict {
    //                             data: exprs,
    //                             map: map.clone().unwrap(),
    //                         }
    //                     } else {
    //                         let dd = DataDict::new(exprs, Some(columns.clone()));
    //                         map = Some(dd.map.clone());
    //                         dd
    //                     };
    //                     let out_e = agg_expr.context_clone();
    //                     // this is safe as we don't return a view on the current context
    //                     // into_owned is important here to guarantee the above
    //                     let current_ctx: Arc<DataDict> =
    //                         Arc::new(unsafe { std::mem::transmute(current_ctx) });
    //                     let o = out_e
    //                         .view_arr(Some(&current_ctx))
    //                         .unwrap()
    //                         .deref()
    //                         .into_owned();
    //                     o
    //                 })
    //                 .collect_trusted();
    //             ArrOk::same_dtype_concat_1d(out)
    //         });
    //         Ok((out.into(), ctx.clone()))
    //     });
    //     self
    // }

    #[cfg(feature = "concat")]
    #[allow(unreachable_patterns)]
    pub fn rolling_apply_with_start(
        &mut self,
        agg_expr: Expr<'a>,
        roll_start: Expr<'a>,
        others: Vec<Expr<'a>>,
        _par: bool,
    ) -> &mut Self {
        use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
        use std::sync::Arc;
        use tea_hash::TpHashMap;
        use tea_lazy::DataDict;
        let name = self.name().unwrap();
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?.deref();
            let others_ref = others
                .par_iter()
                .map(|e| e.view_arr(ctx.as_ref()).unwrap().deref())
                .collect::<Vec<_>>();
            let others_name = others.iter().map(|e| e.name().unwrap()).collect_trusted();
            let roll_start = roll_start.view_arr(ctx.as_ref())?.deref().cast_usize();
            let roll_start_arr = roll_start.view().to_dim1()?;
            let len = arr.len();
            if len != roll_start_arr.len() {
                return Err(format!(
                    "rolling_apply_with_start: arr.len() != roll_start.len(): {} != {}",
                    arr.len(),
                    roll_start_arr.len()
                )
                .into());
            }
            let columns = std::iter::once(name.clone())
                .chain(others_name)
                .collect::<Vec<_>>();
            let mut map: Option<Arc<TpHashMap<String, usize>>> = None;
            let mut agg_expr = agg_expr.flatten();
            agg_expr.simplify();
            let init_data = agg_expr.get_chain_base();
            let nodes = agg_expr.collect_chain_nodes(vec![]);
            let out: ArrOk<'a> = match_arrok!(arr, arr, {
                let arr = arr.view().to_dim1()?;
                let out = if others_ref.is_empty() {
                    zip(roll_start_arr, 0..len)
                        .map(|(mut start, end)| {
                            if start > end {
                                start = end; // the start idx should be inbound
                            }
                            let exprs: Vec<Expr<'_>> =
                                vec![arr.slice(s![start..end + 1]).to_dimd().into()];
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
                            // this is safe as we don't return a view on the current context
                            // into_owned is important here to guarantee the above
                            let current_ctx: Arc<DataDict> = Arc::new(unsafe {
                                std::mem::transmute::<DataDict<'_>, DataDict<'a>>(current_ctx)
                            });
                            let mut data = init_data.clone();
                            let mut ctx = Some(current_ctx);
                            for f in &nodes {
                                (data, ctx) = f((data, ctx)).unwrap();
                            }
                            data.into_arr(ctx).unwrap()
                            // arr0(1).to_dimd().into()
                        })
                        .collect_trusted()
                } else {
                    zip(roll_start_arr, 0..len)
                        .map(|(mut start, end)| {
                            if start > end {
                                start = end; // the start idx should be inbound
                            }
                            let slice = s![start..end + 1];
                            let exprs: Vec<Expr<'_>> =
                                std::iter::once(arr.slice(slice).to_dimd().into())
                                    .chain(others_ref.iter().map(|a| a.slice(slice).into()))
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
                            // this is safe as we don't return a view on the current context
                            // into_owned is important here to guarantee the above
                            let current_ctx: Arc<DataDict> = Arc::new(unsafe {
                                std::mem::transmute::<DataDict<'_>, DataDict<'a>>(current_ctx)
                            });
                            let mut data = init_data.clone();
                            let mut ctx = Some(current_ctx);
                            for f in &nodes {
                                (data, ctx) = f((data, ctx)).unwrap();
                            }
                            data.into_arr(ctx).unwrap()
                        })
                        .collect_trusted()
                };
                ArrOk::same_dtype_concat_1d(out)
            });
            Ok((out.into(), ctx.clone()))
        });
        self
    }

    pub fn get_fix_window_rolling_idx(window: &mut Self, len: Self) -> &mut Self {
        window.chain_f_ctx(move |(arr, ctx)| {
            let window = arr
                .into_arr(ctx.clone())?
                .cast_usize()
                .into_owned()
                .into_scalar()?;
            let len = len
                .view_arr(ctx.as_ref())?
                .deref()
                .cast_usize()
                .into_owned()
                .into_scalar()?;
            let out = if len <= window {
                std::iter::repeat(0).take(len).collect_trusted()
            } else {
                std::iter::repeat(0)
                    .take(window)
                    .chain(1..len - window + 1)
                    .collect_trusted()
            };
            Ok((Arr1::from_vec(out).to_dimd().into(), ctx))
        });
        window
    }

    #[cfg(feature = "time")]
    pub fn get_time_rolling_idx<TD: Into<TimeDelta>>(
        &mut self,
        duration: TD,
        start_by: RollingTimeStartBy,
    ) -> &mut Self {
        let duration: TimeDelta = duration.into();
        self.chain_f_ctx(move |(arr, ctx)| {
            let arr = arr.view_arr(ctx.as_ref())?.deref().cast_datetime_default();
            let view = arr.view().to_dim1()?;
            if view.len() == 0 {
                return Ok((Arr1::from_vec(Vec::<usize>::new()).to_dimd().into(), ctx));
            }
            let out = match start_by {
                // rollling the full duration
                RollingTimeStartBy::Full => {
                    let mut start_time = view[0];
                    let mut start = 0;
                    view.iter()
                        .enumerate()
                        .map(|(i, dt)| {
                            let dt = *dt;
                            if dt < start_time + duration.clone() {
                                start
                            } else {
                                for j in start + 1..=i {
                                    // safety: 0<=j<arr.len()
                                    start_time = unsafe { *view.uget(j) };
                                    if dt < start_time + duration.clone() {
                                        start = j;
                                        break;
                                    }
                                }
                                start
                            }
                        })
                        .collect_trusted()
                }
                // rolling to the start of the duration
                RollingTimeStartBy::DurationStart => {
                    let mut start = 0;
                    let mut dt_truncate = view[0].duration_trunc(duration.clone());
                    view.iter()
                        .enumerate()
                        .map(|(i, dt)| {
                            let dt = *dt;
                            if dt < dt_truncate + duration.clone() {
                                start
                            } else {
                                dt_truncate = dt.duration_trunc(duration.clone());
                                start = i;
                                start
                            }
                        })
                        .collect_trusted()
                }
            };
            Ok((Arr1::from_vec(out).to_dimd().into(), ctx))
        });
        self
    }

    #[cfg(all(feature = "time", feature = "map"))]
    pub fn get_time_rolling_unique_idx<TD: Into<TimeDelta>>(&mut self, duration: TD) -> &mut Self {
        use crate::map::MapExt1d;
        let duration: TimeDelta = duration.into();
        self.chain_f_ctx(move |(arr, ctx)| {
            let arr = arr.view_arr(ctx.as_ref())?.deref().cast_datetime_default();
            let view = arr.view().to_dim1()?;
            if view.len() == 0 {
                return Ok((Arr1::from_vec(Vec::<usize>::new()).to_dimd().into(), ctx));
            }
            let mut start_time = view[0];
            let mut start = 0;
            let out = view
                .iter()
                .enumerate()
                .map(|(i, dt)| {
                    let dt = *dt;
                    let start = if dt < start_time + duration.clone() {
                        start
                    } else {
                        for j in start + 1..=i {
                            // safety: 0<=j<arr.len()
                            start_time = unsafe { *view.uget(j) };
                            if dt < start_time + duration.clone() {
                                start = j;
                                break;
                            }
                        }
                        start
                    };
                    let arr_s = view.slice(s![start..i + 1]);
                    arr_s
                        .wrap()
                        .get_sorted_unique_idx_1d("first".into())
                        .0
                        .into_raw_vec()
                })
                .collect_trusted();
            Ok((Arr1::from_vec(out).to_dimd().into(), ctx))
        });
        self
    }

    #[cfg(feature = "time")]
    pub fn get_time_rolling_offset_idx<TD1: Into<TimeDelta>, TD2: Into<TimeDelta>>(
        &mut self,
        window: TD1,
        offset: TD2,
    ) -> &mut Self {
        let window: TimeDelta = window.into();
        let offset: TimeDelta = offset.into();
        assert!(window >= offset);
        self.chain_f_ctx(move |(data, ctx)| {
            let data = data.view_arr(ctx.as_ref())?.deref().cast_datetime_default();
            let arr = data.view().to_dim1()?;
            if arr.len() == 0 {
                return Ok((Arr1::from_vec(Vec::<usize>::new()).to_dimd().into(), ctx));
            }
            let mut out = vec![vec![]; arr.len()];
            let max_n_offset = window.clone() / offset.clone();
            if max_n_offset < 0 {
                return Err("window // offset < 0!".into());
            }
            (0..arr.len()).for_each(|i| {
                let dt = unsafe { *arr.uget(i) };
                let mut current_n_offset = 0;
                unsafe { out.get_unchecked_mut(i) }.push(i);
                let mut last_dt = dt;
                for j in i + 1..arr.len() {
                    let current_dt = unsafe { *arr.uget(j) };
                    if current_n_offset == max_n_offset && current_dt > dt + window.clone() {
                        break;
                    }

                    let td = current_dt - last_dt;
                    if td < offset.clone() {
                        continue;
                    } else if td == offset.clone() {
                        unsafe { out.get_unchecked_mut(j) }.push(i);
                        current_n_offset += 1;
                        last_dt = dt + offset.clone() * current_n_offset
                    } else {
                        // a large timedelta, need to find the offset
                        if current_dt <= dt + window.clone() {
                            current_n_offset += td / offset.clone();
                            last_dt = dt + offset.clone() * current_n_offset;
                            if current_dt == last_dt {
                                unsafe { out.get_unchecked_mut(j) }.push(i);
                            }
                        }
                    }
                }
            });
            Ok((Arr1::from_vec(out).to_dimd().into(), ctx))
        });
        self
    }

    #[lazy_only(lazy = "rolling_by_startidx", type = "numeric")]
    fn rolling_select_max(&mut self, roll_start: Self) {}
    #[lazy_only(lazy = "rolling_by_vecusize", type = "numeric")]
    fn rolling_select_by_vecusize_max(&mut self, idxs: Self) {}

    #[cfg(feature = "map")]
    #[lazy_only(lazy = "rolling_by_startidx", type = "numeric")]
    fn rolling_select_umax(&mut self, roll_start: Self) {}
    #[cfg(feature = "map")]
    #[lazy_only(lazy = "rolling_by_vecusize", type = "numeric")]
    fn rolling_select_by_vecusize_umax(&mut self, idxs: Self) {}

    #[lazy_only(lazy = "rolling_by_startidx", type = "numeric")]
    fn rolling_select_min(&mut self, roll_start: Self) {}
    #[lazy_only(lazy = "rolling_by_vecusize", type = "numeric")]
    fn rolling_select_by_vecusize_min(&mut self, idxs: Self) {}

    #[cfg(feature = "map")]
    #[lazy_only(lazy = "rolling_by_startidx", type = "numeric")]
    fn rolling_select_umin(&mut self, roll_start: Self) {}
    #[cfg(feature = "map")]
    #[lazy_only(lazy = "rolling_by_vecusize", type = "numeric")]
    fn rolling_select_by_vecusize_umin(&mut self, idxs: Self) {}

    #[lazy_only(lazy = "rolling_by_startidx", type = "numeric")]
    fn rolling_select_median(&mut self, roll_start: Self) {}
    #[lazy_only(lazy = "rolling_by_vecusize", type = "numeric")]
    fn rolling_select_by_vecusize_median(&mut self, idxs: Self) {}

    #[lazy_only(lazy = "rolling_by_startidx", type = "numeric")]
    fn rolling_select_quantile(&mut self, roll_start: Self, q: f64, method: QuantileMethod) {}
    #[lazy_only(lazy = "rolling_by_vecusize", type = "numeric")]
    fn rolling_select_by_vecusize_quantile(&mut self, idxs: Self, q: f64, method: QuantileMethod) {}

    #[lazy_only(lazy = "rolling_by_startidx", type = "nostr")]
    fn rolling_select_first(&mut self, roll_start: Self) {}
    #[lazy_only(lazy = "rolling_by_vecusize", type = "nostr")]
    fn rolling_select_by_vecusize_first(&mut self, idxs: Self) {}

    #[lazy_only(lazy = "rolling_by_startidx", type = "nostr")]
    fn rolling_select_last(&mut self, roll_start: Self) {}
    #[lazy_only(lazy = "rolling_by_vecusize", type = "nostr")]
    fn rolling_select_by_vecusize_last(&mut self, idxs: Self) {}

    #[lazy_only(lazy = "rolling_by_startidx", type = "nostr")]
    fn rolling_select_valid_first(&mut self, roll_start: Self) {}
    #[lazy_only(lazy = "rolling_by_vecusize", type = "nostr")]
    fn rolling_select_by_vecusize_valid_first(&mut self, idxs: Self) {}

    #[lazy_only(lazy = "rolling_by_startidx", type = "nostr")]
    fn rolling_select_valid_last(&mut self, roll_start: Self) {}
    #[lazy_only(lazy = "rolling_by_vecusize", type = "nostr")]
    fn rolling_select_by_vecusize_valid_last(&mut self, idxs: Self) {}

    #[lazy_only(lazy = "rolling_by_startidx", type = "numeric")]
    fn rolling_select_sum(&mut self, roll_start: Self, stable: bool) {}
    #[lazy_only(lazy = "rolling_by_vecusize", type = "numeric")]
    fn rolling_select_by_vecusize_sum(&mut self, idxs: Self, stable: bool) {}

    #[lazy_only(lazy = "rolling_by_startidx", type = "numeric")]
    fn rolling_select_mean(&mut self, roll_start: Self, min_periods: usize, stable: bool) {}
    #[lazy_only(lazy = "rolling_by_vecusize", type = "numeric")]
    fn rolling_select_by_vecusize_mean(&mut self, idxs: Self, min_periods: usize, stable: bool) {}

    #[lazy_only(lazy = "rolling_by_startidx", type = "numeric")]
    fn rolling_select_std(&mut self, roll_start: Self, min_periods: usize, stable: bool) {}
    #[lazy_only(lazy = "rolling_by_vecusize", type = "numeric")]
    fn rolling_select_by_vecusize_std(&mut self, idxs: Self, min_periods: usize, stable: bool) {}

    #[lazy_only(lazy = "rolling_by_startidx", type = "numeric")]
    fn rolling_select_var(&mut self, roll_start: Self, min_periods: usize, stable: bool) {}
    #[lazy_only(lazy = "rolling_by_vecusize", type = "numeric")]
    fn rolling_select_by_vecusize_var(&mut self, idxs: Self, min_periods: usize, stable: bool) {}

    #[lazy_only(lazy = "rolling_by_startidx2", type = "numeric", type2 = "numeric")]
    fn rolling_select_cov(
        &mut self,
        other: Self,
        roll_start: Self,
        min_periods: usize,
        stable: bool,
    ) {
    }
    // #[lazy_only(lazy="rolling_by_vecusize", type="numeric")]
    // fn rolling_select_by_vecusize_var(&mut self, idxs: Self, min_periods: usize, stable: bool) {}

    #[lazy_only(lazy = "rolling_by_startidx2", type = "numeric", type2 = "numeric")]
    fn rolling_select_corr(
        &mut self,
        other: Self,
        roll_start: Self,
        method: CorrMethod,
        min_periods: usize,
        stable: bool,
    ) {
    }
    // #[lazy_only(lazy="rolling_by_vecusize", type="numeric")]
    // fn rolling_select_by_vecusize_var(&mut self, idxs: Self, min_periods: usize, stable: bool) {}

    #[lazy_only(lazy = "rolling_by_startidx2", type = "numeric", type2 = "numeric")]
    fn rolling_select_weight_mean(
        &mut self,
        other: Self,
        roll_start: Self,
        min_periods: usize,
        stable: bool,
    ) {
    }

    #[lazy_only(lazy = "rolling_by_startidx2", type = "numeric", type2 = "bool")]
    fn rolling_select_cut_mean(&mut self, other: Self, roll_start: Self, min_periods: usize) {}
}
