use crate::datatype::{DateTime, TimeDelta};
use crate::lazy::{Expr, ExprElement, RefType};
use crate::{Arr1, ArrView1, CollectTrustedToVec, Number, WrapNdarray};
use ndarray::s;
use std::iter::zip;

impl<'a, T: ExprElement + 'a> Expr<'a, T> {
    // pub fn rolling_by_time<TD: Into<TimeDelta>>(self, by: Expr<'a, DateTime>, duration: TD) -> Expr<'a, f64> {
    //     self.chain_view_f(move |arr| {
    //         let duration: TimeDelta = duration.into();
    //         let arr = arr.to_dim1()?;
    //         let by = by.eval()?;
    //         let by_arr = by.view_arr().to_dim1()?;

    //         Ok(arr.to_owned())
    //     },
    //     RefType::False)
    // }

    pub fn rolling_select_agg<F, U>(self, roll_start: Expr<'a, usize>, f: F) -> Expr<'a, U>
    where
        U: ExprElement + 'a,
        F: Fn(&ArrView1<T>) -> U + Send + Sync + 'a,
    {
        self.chain_view_f(
            move |arr| {
                let arr = arr.to_dim1()?;
                let roll_start = roll_start.eval()?;
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
                Ok(out.into())
            },
            RefType::False,
        )
    }

    pub fn rolling_select_max(self, roll_start: Expr<'a, usize>) -> Expr<'a, T>
    where
        T: Number,
    {
        self.rolling_select_agg(roll_start, |arr| arr.max_1d())
    }

    pub fn rolling_select_min(self, roll_start: Expr<'a, usize>) -> Expr<'a, T>
    where
        T: Number,
    {
        self.rolling_select_agg(roll_start, |arr| arr.min_1d())
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

    pub fn rolling_select_mean(self, roll_start: Expr<'a, usize>) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.rolling_select_agg(roll_start, |arr| arr.mean_1d(false))
    }

    pub fn rolling_select_sum(self, roll_start: Expr<'a, usize>) -> Expr<'a, T>
    where
        T: Number,
    {
        self.rolling_select_agg(roll_start, |arr| arr.sum_1d(false))
    }

    pub fn rolling_select_std(self, roll_start: Expr<'a, usize>) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.rolling_select_agg(roll_start, |arr| arr.std_1d(false))
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
}
