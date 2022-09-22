use super::prelude::*;

impl<T, S, D> ArrBase<S, D>
where
    S: Data<Elem = T>,
    D: Dimension,
{
    /// Remove NaN values in 1d array.
    #[inline]
    pub fn remove_nan_1d(self) -> Arr1<T>
    where
        D: Dim1,
        T: Number,
    {
        Arr1::from_iter(self.into_iter().filter(|v| v.notnan()))
    }

    impl_map_nd!(
        zscore,
        /// Sandardize the array using zscore method on a given axis
        #[inline]
        pub fn zscore_1d<S2>(&self, out: &mut ArrBase<S2, D>, stable: bool) -> f64
        {where T: Number}
        {
            let (mean, var) = self.meanvar_1d(stable);
            if var == 0. {
                out.apply_mut(|v| *v = 0.);
            } else if var.isnan() {
                out.apply_mut(|v| *v = f64::NAN);
            } else {
                out.apply_mut_with(self, |vo, v| *vo = (v.f64() - mean) / var.sqrt());
            }
        }
    );

    impl_map_nd!(
        clip,
        pub fn clip_1d<S2, T2, T3>(&self, out: &mut ArrBase<S2, D>, min: T2, max: T3) -> T
        {
            where
            T: Number, T2: Number; AsPrimitive<T>, T3: Number; AsPrimitive<T>
        }
        {
            let (min, max) = (T::fromas(min), T::fromas(max));
            assert!(min <= max, "min must smaller than max in clamp");
            assert!(
                min.notnan() & max.notnan(),
                "min and max should not be NaN in clamp"
            );
            out.apply_mut_with(self, |vo, v| {
                if *v > max {
                    // Note that NaN is excluded
                    *vo = max
                } else if *v < min {
                    *vo = min;
                } else {
                    *vo = *v
                }
            })
        }
    );

    impl_map_nd!(
        fillna,
        pub fn fillna_1d<S2, T2>(&self, out: &mut ArrBase<S2, D>, method: Option<&str>, value: Option<T2>) -> T
        {
            where
            T: Number,
            f64: AsPrimitive<T>,
            T2: AsPrimitive<T>; Send; Sync
        }
        {
            if let Some(method) = method {
                let mut last_valid: Option<T> = None;
                let mut f = |vo: &mut T, v: &T| {
                    if v.isnan() {
                        if let Some(lv) = last_valid {
                            *vo = lv;
                        } else {
                            *vo = f64::NAN.as_();
                        }
                    } else { // v is valid, update last_valid
                        *vo = *v;
                        last_valid = Some(*v);
                    }
                };
                if method == "ffill" {
                    out.apply_mut_with(self, f)
                } else if method == "bfill" {
                    for (vo, v) in zip(out, self).rev() {
                        f(vo, v);
                    }
                } else {
                    panic!("Not support method: {} in fillna_inplace", method);
                }
            } else {
                let value = value.expect("Fill value must be pass when using value to fillna");
                let value: T = value.as_();
                out.apply_mut_with(self, |vo, v| if v.isnan() {
                    *vo = value;
                } else {
                    *vo = *v;
                });
            }
        }
    );

    impl_map_nd!(
        winsorize,
        pub fn winsorize_1d<S2>(&self, out: &mut ArrBase<S2, D>, method: Option<&str>, method_params: Option<f64>, stable: bool) -> T
        {
            where
            T: Number,
            f64: AsPrimitive<T>
        }
        {
            let method = method.unwrap_or("quantile");
            if method == "quantile" {
                // default method is clip 1% and 99% quantile
                let method_params = method_params.unwrap_or(0.01);
                let min = self.quantile_1d(method_params, None);
                let max = self.quantile_1d(1. - method_params, None);
                if min.notnan() && (min != max) {
                    self.clip_1d(out, min, max);
                } else {
                    // elements in the given axis are all NaN or equal to a constant
                    self.clone_to(out);
                }
            } else if method == "median" {
                // default method is clip median - 3 * mad, median + 3 * mad
                let method_params = method_params.unwrap_or(3.);
                let median = self.median_1d();
                if median.notnan() {
                    let mad = self.mapv(|v| (v.f64() - median).abs()).median_1d();
                    let min = median - method_params * mad;
                    let max = median + method_params * mad;
                    self.clip_1d(out, min, max);
                } else {
                    self.clone_to(out);
                }
            } else if method == "sigma" {
                 // default method is clip mean - 3 * std, mean + 3 * std
                let method_params = method_params.unwrap_or(3.);
                let (mean, var) = self.meanvar_1d(stable);
                if mean.notnan() {
                    let std = var.sqrt();
                    let min = mean - method_params * std;
                    let max = mean + method_params * std;
                    self.clip_1d(out, min, max);
                } else {
                    self.clone_to(out);
                }
            } else {
                panic!("Not support {} method in winsorize", method);
            }
        }
    );

    impl_map_nd!(
        argsort,
        /// count not NaN number of an array on a given axis
        pub fn argsort_1d<S2>(&self, out: &mut ArrBase<S2, D>) -> i32 {
            where
            T: Number
        }
        {
            assert!(out.len() >= self.len());
            let mut i = 0;
            out.apply_mut(|v| {
                *v = i;
                i += 1;
            }); // set elements of out array
            out.sort_unstable_by(|a, b| {
                let (va, vb) = unsafe { (*self.uget((*a) as usize), *self.uget((*b) as usize)) }; // safety: out不超过self的长度
                va.nan_sort_cmp(&vb)
            });
        }
    );

    impl_map_nd!(
        rank,
        /// rank the array in a given axis
        #[allow(unused_assignments)]
        pub fn rank_1d<S2>(&self, out: &mut ArrBase<S2, D>, pct: bool) -> f64 {
            where
            T: Number
        }
        {
            let len = self.len();
            assert!(
                out.len() >= len,
                "the length of the input array not equal to the length of the output array"
            );
            if len == 0 {
                return;
            } else if len == 1 {
                // safety: out.len() == self.len() == 1
                unsafe { *out.uget_mut(0) = 1. };
                return;
            }
            // argsort at first
            let mut idx_sorted = Arr1::from_iter(0..len);
            idx_sorted.sort_unstable_by(|a, b| {
                let (va, vb) = unsafe { (*self.uget((*a) as usize), *self.uget((*b) as usize)) }; // safety: out不超过self的长度
                va.nan_sort_cmp(&vb)
            });
            // if the smallest value is nan then all the elements are nan
            if unsafe { *self.uget(*idx_sorted.uget(0)) }.isnan() {
                return out.apply_mut(|v| *v = f64::NAN);
            }
            let mut repeat_num = 1usize;
            let mut nan_flag = false;
            let (mut cur_rank, mut sum_rank) = (1usize, 0usize);
            let (mut idx, mut idx1) = (0, 0);
            if !pct {
                unsafe {
                    for i in 0..len - 1 {
                        // safe because i_max = self.len()-2 and self.len() >= 2
                        (idx, idx1) = (*idx_sorted.uget(i), *idx_sorted.uget(i + 1));
                        let (v, v1) = (*self.uget(idx), *self.uget(idx1)); // 下一个值，safe because idx1 < arr.len()
                        if v1.isnan() {
                            // 下一个值是nan，说明后面的值全是nan
                            sum_rank += cur_rank;
                            cur_rank += 1;
                            for j in 0..repeat_num {
                                // safe because i >= repeat_num
                                *out.uget_mut(*idx_sorted.uget(i - j)) =
                                    sum_rank.f64() / repeat_num.f64()
                            }
                            idx = i + 1;
                            nan_flag = true;
                            break;
                        } else if v == v1 {
                            // 当前值和下一个值相同，说明开始重复
                            repeat_num += 1;
                            sum_rank += cur_rank;
                            cur_rank += 1;
                        } else if repeat_num == 1 {
                            // 无重复，可直接得出排名
                            *out.uget_mut(idx) = cur_rank as f64;
                            cur_rank += 1;
                        } else {
                            // 当前元素是最后一个重复元素
                            sum_rank += cur_rank;
                            cur_rank += 1;
                            for j in 0..repeat_num {
                                // safe because i >= repeat_num
                                *out.uget_mut(*idx_sorted.uget(i - j)) =
                                    sum_rank.f64() / repeat_num.f64()
                            }
                            sum_rank = 0; // rank和归零
                            repeat_num = 1; // 重复计数归一
                        }
                    }
                    if nan_flag {
                        for i in idx..len {
                            *out.uget_mut(*idx_sorted.uget(i)) = f64::NAN;
                        }
                    } else {
                        sum_rank += cur_rank;
                        for i in len - repeat_num..len {
                            // safe because repeat_num <= len
                            *out.uget_mut(*idx_sorted.uget(i)) = sum_rank.f64() / repeat_num.f64()
                        }
                    }
                }
            } else {
                let notnan_count = self.count_notnan_1d();
                unsafe {
                    for i in 0..len - 1 {
                        // safe because i_max = arr.len()-2 and arr.len() >= 2
                        (idx, idx1) = (*idx_sorted.uget(i), *idx_sorted.uget(i + 1));
                        let (v, v1) = (*self.uget(idx), *self.uget(idx1)); // 下一个值，safe because idx1 < arr.len()
                        if v1.isnan() {
                            // 下一个值是nan，说明后面的值全是nan
                            sum_rank += cur_rank;
                            cur_rank += 1;
                            for j in 0..repeat_num {
                                // safe because i >= repeat_num
                                *out.uget_mut(*idx_sorted.uget(i - j)) =
                                    sum_rank.f64() / (repeat_num * notnan_count).f64()
                            }
                            idx = i + 1;
                            nan_flag = true;
                            break;
                        } else if v == v1 {
                            // 当前值和下一个值相同，说明开始重复
                            repeat_num += 1;
                            sum_rank += cur_rank;
                            cur_rank += 1;
                        } else if repeat_num == 1 {
                            // 无重复，可直接得出排名
                            *out.uget_mut(idx) = cur_rank.f64() / notnan_count.f64();
                            cur_rank += 1;
                        } else {
                            // 当前元素是最后一个重复元素
                            sum_rank += cur_rank;
                            cur_rank += 1;
                            for j in 0..repeat_num {
                                // safe because i >= repeat_num
                                *out.uget_mut(*idx_sorted.uget(i - j)) =
                                    sum_rank.f64() / (repeat_num * notnan_count).f64()
                            }
                            sum_rank = 0; // rank和归零
                            repeat_num = 1; // 重复计数归一
                        }
                    }
                    if nan_flag {
                        for i in idx..len {
                            *out.uget_mut(*idx_sorted.uget(i)) = f64::NAN;
                        }
                    } else {
                        sum_rank += cur_rank;
                        for i in len - repeat_num..len {
                            // safe because repeat_num <= len
                            *out.uget_mut(*idx_sorted.uget(i)) =
                                sum_rank.f64() / (repeat_num * notnan_count).f64()
                        }
                    }
                }
            }
        }
    );
}

impl<T, S, D> ArrBase<S, D>
where
    S: DataMut<Elem = T>,
    D: Dimension,
{
    impl_map_inplace_nd!(
        fillna_inplace,
        pub fn fillna_inplace_1d<T2>(&mut self, method: Option<&str>, value: Option<T2>)
        {
            where
            T: Number,
            T2: AsPrimitive<T>; Send; Sync
        }
        {
            if let Some(method) = method {
                let mut last_valid: Option<T> = None;
                let mut f = |v: &mut T| {
                    if v.isnan() {
                        if let Some(lv) = last_valid {
                            *v = lv;
                        }
                    } else { // v is valid, update last_valid
                        last_valid = Some(*v);
                    }
                };
                if method == "ffill" {
                    self.apply_mut(f)
                } else if method == "bfill" {
                    for v in self.iter_mut().rev() {
                        f(v);
                    }
                } else {
                    panic!("Not support method: {} in fillna_inplace", method);
                }
            } else {
                let value = value.expect("Fill value must be pass when using value to fillna");
                let value: T = value.as_();
                self.apply_mut(|v| if v.isnan() {
                    *v = value
                });
            }
        }
    );

    impl_map_inplace_nd!(
        clip_inplace,
        pub fn clip_inplace_1d<T2, T3>(&mut self, min: T2, max: T3)
        {
            where
            T: Number,
            T2: Number; AsPrimitive<T>,
            T3: Number; AsPrimitive<T>,
        }
        {
            let (min, max) = (T::fromas(min), T::fromas(max));
            assert!(min <= max, "min must smaller than max in clamp");
            assert!(
                min.notnan() & max.notnan(),
                "min and max should not be NaN in clamp"
            );
            self.apply_mut(|v| {
                if *v > max {
                    // Note that NaN is excluded
                    *v = max;
                } else if *v < min {
                    *v = min;
                }
            })
        }
    );

    impl_map_inplace_nd!(
        zscore_inplace,
        /// Sandardize the array using zscore method on a given axis
        #[inline]
        pub fn zscore_inplace_1d(&mut self, stable: bool) {
        where
            T: Number,
            f64: AsPrimitive<T>
        }
        {
            let (mean, var) = self.meanvar_1d(stable);
            if var == 0. {
                self.apply_mut(|v| *v = 0.0.as_());
            } else if var.isnan() {
                self.apply_mut(|v| *v = f64::NAN.as_());
            } else {
                self.apply_mut(|v| *v = ((v.f64() - mean) / var.sqrt()).as_());
            }
        }
    );

    impl_map_inplace_nd!(
        winsorize_inplace,
        pub fn winsorize_inplace_1d(&mut self, method: Option<&str>, method_params: Option<f64>, stable: bool)
        {
            where
            T: Number,
            f64: AsPrimitive<T>
        }
        {
            let method = method.unwrap_or("quantile");
            if method == "quantile" {
                // default method is clip 1% and 99% quantile
                let method_params = method_params.unwrap_or(0.01);
                let min = self.quantile_1d(method_params, None);
                let max = self.quantile_1d(1. - method_params, None);
                if min.notnan() && (min != max) {
                    self.clip_inplace_1d(min, max);
                }
            } else if method == "median" {
                // default method is clip median - 3 * mad, median + 3 * mad
                let method_params = method_params.unwrap_or(3.);
                let median = self.median_1d();
                if median.notnan() {
                    let mad = self.mapv(|v| (v.f64() - median).abs()).median_1d();
                    let min = median - method_params * mad;
                    let max = median + method_params * mad;
                    self.clip_inplace_1d(min, max);
                }
            } else if method == "sigma" {
                 // default method is clip mean - 3 * std, mean + 3 * std
                let method_params = method_params.unwrap_or(3.);
                let (mean, var) = self.meanvar_1d(stable);
                if mean.notnan() {
                    let std = var.sqrt();
                    let min = mean - method_params * std;
                    let max = mean + method_params * std;
                    self.clip_inplace_1d(min, max);
                }
            } else {
                panic!("Not support {} method in winsorize", method);
            }
        }
    );
}
