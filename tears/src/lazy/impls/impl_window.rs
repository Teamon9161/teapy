use crate::datatype::{DateTime, TimeDelta};
use crate::lazy::{Expr, ExprElement, RefType};
use crate::{Arr1, ArrView1, CollectTrustedToVec, Number, WrapNdarray};
use ndarray::s;
use std::iter::zip;

macro_rules! impl_rolling_by_time_agg {
    ($(@$func: ident-$func1: ident, $agg_func: ident ($($p:ident),*) -> $T: ty),* ,) => {
        impl<'a, T: ExprElement + 'a> Expr<'a, T> {
            $(
                pub fn $func(self, roll_start: Expr<'a, usize>) -> Expr<'a, $T>
                where
                    T: Number,
                {
                    self.rolling_select_agg(roll_start, |arr| arr.$agg_func($($p),*))
                }

                pub fn $func1(self, idxs: Expr<'a, Vec<usize>>) -> Expr<'a, $T>
                where
                    T: Number,
                {
                    self.rolling_select_agg_by_vecusize(idxs, |arr| arr.$agg_func($($p),*))
                }
            )*
        }
    };
}

impl_rolling_by_time_agg!(
    @rolling_select_max-rolling_select_by_vecusize_max, max_1d() -> T,
    @rolling_select_min-rolling_select_by_vecusize_min, min_1d() -> T,
    @rolling_select_mean-rolling_select_by_vecusize_mean, mean_1d(false) -> f64,
    @rolling_select_sum-rolling_select_by_vecusize_sum, sum_1d(false) -> T,
    @rolling_select_std-rolling_select_by_vecusize_std, std_1d(false) -> f64,
);

impl<'a, T: ExprElement + 'a> Expr<'a, T> {
    pub fn rolling_select_agg_by_vecusize<F, U>(
        self,
        idxs: Expr<'a, Vec<usize>>,
        f: F,
    ) -> Expr<'a, U>
    where
        F: Fn(&ArrView1<T>) -> U + Send + Sync + 'a,
        U: ExprElement + 'a,
        T: Clone,
    {
        self.chain_view_f_ct(
            move |(arr, ctx)| {
                let arr = arr.to_dim1()?;
                let (idxs, ctx) = idxs.eval(ctx)?;
                let idxs_arr = idxs.view_arr().to_dim1()?;
                let out = idxs_arr
                    .into_iter()
                    .map(|idx| {
                        let mut out = Arr1::default(idx.len());
                        let slc = ArrView1::from_ref_vec(idx.len(), idx);
                        unsafe { arr.take_clone_1d_unchecked(out.view_mut(), slc) }
                        f(&out.view())
                    })
                    .collect_trusted();
                let out = Arr1::from_vec(out).to_dimd();
                Ok((out.into(), ctx))
            },
            RefType::False,
        )
    }

    pub fn rolling_select_agg<F, U>(self, roll_start: Expr<'a, usize>, f: F) -> Expr<'a, U>
    where
        U: ExprElement + 'a,
        F: Fn(&ArrView1<T>) -> U + Send + Sync + 'a,
    {
        self.chain_view_f_ct(
            move |(arr, ct)| {
                let arr = arr.to_dim1()?;
                let (roll_start, ct) = roll_start.eval(ct)?;
                let roll_start_arr = roll_start.view_arr().to_dim1()?;
                if arr.len() != roll_start_arr.len() {
                    return Err(format!(
                        "rolling_select_agg: arr.len() != roll_start.len(): {} != {}",
                        arr.len(),
                        roll_start_arr.len()
                    )
                    .into());
                }
                let len = arr.len();
                let out = zip(roll_start_arr, 0..len)
                    .map(|(start, end)| {
                        let current_arr = arr.slice(s![start..end + 1]).wrap();
                        f(&current_arr)
                    })
                    .collect_trusted();
                let out = Arr1::from_vec(out).to_dimd();
                Ok((out.into(), ct))
            },
            RefType::False,
        )
    }

    pub fn rolling_select_umax(self, roll_start: Expr<'a, usize>) -> Expr<'a, T>
    where
        T: Number,
    {
        self.rolling_select_agg(roll_start, |arr| arr.sorted_unique_1d().max_1d())
    }

    pub fn rolling_select_umin(self, roll_start: Expr<'a, usize>) -> Expr<'a, T>
    where
        T: Number,
    {
        self.rolling_select_agg(roll_start, |arr| arr.sorted_unique_1d().min_1d())
    }
}

pub enum RollingTimeStartBy {
    Full,
    DurationStart,
}

impl<'a> Expr<'a, DateTime> {
    /// Rolling with a timedelta, note that the time should be sorted before rolling.
    ///
    /// this function return start index of each window.
    pub fn get_time_rolling_idx<TD: Into<TimeDelta>>(
        self,
        duration: TD,
        start_by: RollingTimeStartBy,
    ) -> Expr<'a, usize> {
        let duration: TimeDelta = duration.into();
        self.chain_view_f(
            move |arr| {
                if arr.len() == 0 {
                    return Ok(Arr1::from_vec(vec![]).to_dimd().into());
                }
                let arr = arr.to_dim1()?;
                let out = match start_by {
                    // rollling the full duration
                    RollingTimeStartBy::Full => {
                        let mut start_time = arr[0];
                        let mut start = 0;
                        arr.iter()
                            .enumerate()
                            .map(|(i, dt)| {
                                let dt = *dt;
                                if dt < start_time + duration.clone() {
                                    start
                                } else {
                                    for j in start + 1..=i {
                                        // safety: 0<=j<arr.len()
                                        start_time = unsafe { *arr.uget(j) };
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
                        let mut dt_truncate = arr[0].duration_trunc(duration.clone());
                        arr.iter()
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

                Ok(Arr1::from_vec(out).to_dimd().into())
            },
            RefType::False,
        )
    }

    /// Rolling with a timedelta, note that the time should be sorted before rolling.
    ///
    /// this function return start index of each window.
    pub fn get_time_rolling_unique_idx<TD: Into<TimeDelta>>(
        self,
        duration: TD,
    ) -> Expr<'a, Vec<usize>> {
        let duration: TimeDelta = duration.into();
        self.chain_view_f(
            move |arr| {
                if arr.len() == 0 {
                    return Ok(Arr1::from_vec(vec![]).to_dimd().into());
                }
                let arr = arr.to_dim1()?;
                // .expect("rolling only support array with dim 1");
                let mut start_time = arr[0];
                let mut start = 0;
                let out = arr
                    .iter()
                    .enumerate()
                    .map(|(i, dt)| {
                        let dt = *dt;
                        let start = if dt < start_time + duration.clone() {
                            start
                        } else {
                            for j in start + 1..=i {
                                // safety: 0<=j<arr.len()
                                start_time = unsafe { *arr.uget(j) };
                                if dt < start_time + duration.clone() {
                                    start = j;
                                    break;
                                }
                            }
                            start
                        };
                        let arr_s = arr.slice(s![start..i + 1]);
                        arr_s
                            .wrap()
                            .get_sorted_unique_idx_1d("first".into())
                            .0
                            .into_raw_vec()
                    })
                    .collect_trusted();
                Ok(Arr1::from_vec(out).to_dimd().into())
            },
            RefType::False,
        )
    }

    /// Rolling with a timedelta, get the idx of the time on the same offset.
    /// The time expression should be sorted before rolling.
    pub fn get_time_rolling_offset_idx<TD1: Into<TimeDelta>, TD2: Into<TimeDelta>>(
        self,
        window: TD1,
        offset: TD2,
    ) -> Expr<'a, Vec<usize>> {
        let window: TimeDelta = window.into();
        let offset: TimeDelta = offset.into();
        assert!(window >= offset);
        self.chain_view_f(
            move |arr| {
                if arr.len() == 0 {
                    return Ok(Arr1::from_vec(vec![]).to_dimd().into());
                }
                let arr = arr.to_dim1()?;
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
                let out = Arr1::from_vec(out);
                Ok(out.to_dimd().into())
            },
            RefType::False,
        )
    }
}
