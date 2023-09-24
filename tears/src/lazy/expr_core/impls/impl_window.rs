use std::iter::zip;
use std::sync::Arc;

use super::export::*;
use crate::lazy::DataDict;
use crate::{Arr1, CollectTrustedToVec, TimeDelta};
use ahash::HashMap;
use ndarray::s;

pub enum RollingTimeStartBy {
    Full,
    DurationStart,
}

macro_rules! impl_rolling_by_startidx_agg {
    ($func_name: ident, $func_1d: ident ($($p: ident: $ty: ty),*) $(.$func2: ident ($($p2: ident: $ty2: ty),*))*) => {
        impl<'a> Expr<'a> {
            pub fn $func_name(&mut self, roll_start: Expr<'a>  $(,$p:$ty)* $($(,$p2:$ty2)*)*) -> &mut Self
            {
                self.chain_f_ctx(
                    move |(data, ctx)| {
                        let arr = data.view_arr(ctx.as_ref())?.deref();
                        let roll_start = roll_start.view_arr(ctx.as_ref())?.deref().cast_usize();
                        let roll_start_arr = roll_start.view().to_dim1()?;
                        let len = arr.len();
                        if len != roll_start_arr.len() {
                            return Err(format!(
                                "rolling_select_agg: arr.len() != roll_start.len(): {} != {}",
                                arr.len(),
                                roll_start_arr.len()
                            )
                            .into());
                        }

                        let out: ArrOk<'a> = match_arrok!(numeric arr, arr, {
                            let arr = arr.view().to_dim1()?;
                            let out = zip(roll_start_arr, 0..len)
                            .map(|(mut start, end)| {
                                if start > end {
                                    start = end;  // the start idx should be inbound
                                }
                                let current_arr = arr.slice(s![start..end + 1]).wrap();
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
}

impl_rolling_by_startidx_agg!(rolling_select_max, max_1d());
impl_rolling_by_startidx_agg!(rolling_select_min, min_1d());
impl_rolling_by_startidx_agg!(rolling_select_mean, mean_1d(stable: bool));
impl_rolling_by_startidx_agg!(rolling_select_sum, sum_1d(stable: bool));
impl_rolling_by_startidx_agg!(rolling_select_std, std_1d(stable: bool));
impl_rolling_by_startidx_agg!(rolling_select_var, var_1d(stable: bool));
impl_rolling_by_startidx_agg!(rolling_select_umax, sorted_unique_1d().max_1d());
impl_rolling_by_startidx_agg!(rolling_select_umin, sorted_unique_1d().min_1d());

impl<'a> Expr<'a> {
    #[allow(unreachable_patterns)]
    pub fn rolling_apply_with_start(
        &mut self,
        agg_expr: Expr<'a>,
        roll_start: Expr<'a>,
        others: Vec<Expr<'a>>,
        _par: bool,
    ) -> &mut Self {
        let name = self.name().unwrap();
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?.deref();
            let others_ref = others
                .iter()
                .map(|e| e.view_arr(ctx.as_ref()).unwrap().deref())
                .collect_trusted();
            let others_name = others.iter().map(|e| e.name().unwrap()).collect_trusted();
            let roll_start = roll_start.view_arr(ctx.as_ref())?.deref().cast_usize();
            let roll_start_arr = roll_start.view().to_dim1()?;
            let len = arr.len();
            if len != roll_start_arr.len() {
                return Err(format!(
                    "rolling_select_agg: arr.len() != roll_start.len(): {} != {}",
                    arr.len(),
                    roll_start_arr.len()
                )
                .into());
            }
            let columns = std::iter::once(name.clone())
                .chain(others_name)
                .collect::<Vec<_>>();
            let mut map: Option<Arc<HashMap<String, usize>>> = None;
            let agg_expr = agg_expr.flatten();
            let out: ArrOk<'a> = match_arrok!(arr, arr, {
                let arr = arr.view().to_dim1()?;
                let out = zip(roll_start_arr, 0..len)
                    .map(|(mut start, end)| {
                        if start > end {
                            start = end; // the start idx should be inbound
                        }
                        let current_arr = arr.slice(s![start..end + 1]).wrap();
                        let current_others: Vec<ArrOk> = others_ref
                            .iter()
                            .map(|arr| arr.slice(s![start..=end]))
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

    pub fn get_time_rolling_unique_idx<TD: Into<TimeDelta>>(&mut self, duration: TD) -> &mut Self {
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
}
