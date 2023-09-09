use super::super::export::*;

impl_map_nd!(
    ts_argmin,
    pub fn ts_argmin_1d<S2>(
        &self,
        out: &mut ArrBase<S2, D>,
        window: usize,
        min_periods: usize,
    ) -> f64
    {where T: Number,}
    {
        let window = min(self.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {v.write(f64::NAN);});
        }
        let mut min: T = T::max_();
        let mut min_idx = 0usize;
        let mut n = 0usize;
        for i in 0..window - 1 {
            // 安全性：i不会超过arr和out的长度
            let v = unsafe { *self.uget(i) };
            if v.notnan() {
                n += 1
            };
            if v < min {
                (min, min_idx) = (v, i);
            }
            let out = unsafe {out.uget_mut(i)};
            if n >= min_periods {
                out.write((min_idx + 1).f64());
            } else {
                out.write(f64::NAN);
            };
        }
        for (start, end) in (window - 1..self.len()).enumerate() {
            // 安全性：start和end不会超过self和out的长度
            unsafe {
                let v = *self.uget(end);
                if v.notnan() {
                    n += 1
                };
                if min_idx < start {
                    // 最小值已经失效，重新找到最小值
                    let v = *self.uget(start);
                    min = if v.isnan() { T::max_() } else { v };
                    for i in start..=end {
                        let v = *self.uget(i);
                        if v <= min {
                            (min, min_idx) = (v, i);
                        }
                    }
                } else if v <= min {
                    // 当前是最小值
                    (min, min_idx) = (v, end);
                }
                let out = out.uget_mut(end);
                if n >= min_periods {
                    out.write((min_idx - start + 1).f64());
                } else {
                    out.write(f64::NAN);
                };
                if self.uget(start).notnan() {
                    n -= 1
                };
            }
        }
    }
);

impl_map_nd!(
    ts_argmax,
    pub fn ts_argmax_1d<S2>(
        &self,
        out: &mut ArrBase<S2, D>,
        window: usize,
        min_periods: usize,
    ) -> f64
    {where T: Number,}
    {
        let window = min(self.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {v.write(f64::NAN);});
        }
        let mut max: T = T::min_();
        let mut max_idx = 0usize;
        let mut n = 0usize;
        for i in 0..window - 1 {
            // 安全性：i不会超过arr和out的长度
            let v = unsafe { *self.uget(i) };
            if v.notnan() {
                n += 1
            };
            if v > max {
                (max, max_idx) = (v, i);
            }
            let out = unsafe {out.uget_mut(i)};
            if n >= min_periods {
                out.write((max_idx + 1).f64());
            } else {
                out.write(f64::NAN);
            };
        }
        for (start, end) in (window - 1..self.len()).enumerate() {
            // 安全性：start和end不会超过self和out的长度
            unsafe {
                let v = *self.uget(end);
                if v.notnan() {
                    n += 1
                };
                if max_idx < start {
                    // 最大值已经失效，重新找到最大值
                    let v = *self.uget(start);
                    max = if v.isnan() { T::min_() } else { v };
                    for i in start..=end {
                        let v = *self.uget(i);
                        if v >= max {
                            (max, max_idx) = (v, i);
                        }
                    }
                } else if v >= max {
                    // 当前是最大值
                    (max, max_idx) = (v, end);
                }
                let out = out.uget_mut(end);
                if n >= min_periods {
                    out.write((max_idx - start + 1).f64());
                } else {
                    out.write(f64::NAN);
                };
                if self.uget(start).notnan() {
                    n -= 1
                };
            }
        }
    }
);

impl_map_nd!(
    ts_min,
    pub fn ts_min_1d<S2>(
        &self,
        out: &mut ArrBase<S2, D>,
        window: usize,
        min_periods: usize,
    ) -> f64
    {where T: Number,}
    {
        let window = min(self.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {v.write(f64::NAN);});
        }
        let mut min: T = T::max_();
        let mut min_idx = 0usize;
        let mut n = 0usize;
        for i in 0..window - 1 {
            // 安全性：i不会超过arr和out的长度
            let v = unsafe { *self.uget(i) };
            if v.notnan() {
                n += 1
            };
            if v < min {
                (min, min_idx) = (v, i);
            }
            let out = unsafe {out.uget_mut(i)};
            if n >= min_periods {
                out.write(min.f64());
            } else {
                out.write(f64::NAN);
            };
        }
        for (start, end) in (window - 1..self.len()).enumerate() {
            // 安全性：start和end不会超过self和out的长度
            unsafe {
                let v = *self.uget(end);
                if v.notnan() {
                    n += 1
                };
                if min_idx < start {
                    // 最小值已经失效，重新找到最小值
                    let v = *self.uget(start);
                    min = if v.isnan() { T::max_() } else { v };
                    for i in start..=end {
                        let v = *self.uget(i);
                        if v <= min {
                            (min, min_idx) = (v, i);
                        }
                    }
                } else if v <= min {
                    // 当前是最小值
                    (min, min_idx) = (v, end);
                }
                let out = out.uget_mut(end);
                if n >= min_periods {
                    out.write(min.f64());
                } else {
                    out.write(f64::NAN);
                };
                if self.uget(start).notnan() {
                    n -= 1
                };
            }
        }
    }
);

impl_map_nd!(
    ts_max,
    pub fn ts_max_1d<S2>(
        &self,
        out: &mut ArrBase<S2, D>,
        window: usize,
        min_periods: usize,
    ) -> f64
    {where T: Number,}
    {
        let window = min(self.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {v.write(f64::NAN);});
        }
        let mut max: T = T::min_();
        let mut max_idx = 0usize;
        let mut n = 0usize;
        for i in 0..window - 1 {
            // 安全性：i不会超过arr和out的长度
            let v = unsafe { *self.uget(i) };
            if v.notnan() {
                n += 1
            };
            if v > max {
                (max, max_idx) = (v, i);
            }
            let out = unsafe {out.uget_mut(i)};
            if n >= min_periods {
                out.write(max.f64());
            } else {
                out.write(f64::NAN);
            };

        }
        for (start, end) in (window - 1..self.len()).enumerate() {
            // 安全性：start和end不会超过self和out的长度
            unsafe {
                let v = *self.uget(end);
                if v.notnan() {
                    n += 1
                };
                if max_idx < start {
                    // 最大值已经失效，重新找到最大值
                    let v = *self.uget(start);
                    max = if v.isnan() { T::min_() } else { v };
                    for i in start..=end {
                        let v = *self.uget(i);
                        if v >= max {
                            (max, max_idx) = (v, i);
                        }
                    }
                } else if v >= max {
                    // 当前是最大值
                    (max, max_idx) = (v, end);
                }
                let out = out.uget_mut(end);
                if n >= min_periods {
                    out.write(max.f64());
                } else {
                    out.write(f64::NAN);
                };
                if self.uget(start).notnan() {
                    n -= 1
                };
            }
        }
    }
);

impl_map_nd!(
    ts_rank,
    pub fn ts_rank_1d<S2>(
        &self,
        out: &mut ArrBase<S2, D>,
        window: usize,
        min_periods: usize,
    ) -> f64
    {where T: Number,}
    {
        let window = min(self.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {v.write(f64::NAN);});
        }
        let mut n = 0usize;
        for i in 0..window - 1 {
            // 安全性：i不会超过arr和out的长度
            unsafe {
                let v = *self.uget(i);
                let mut n_repeat = 1; // 当前值的重复次数
                let mut rank = 1.; // 先假设为第一名，每当有元素比他更小，排名就加1
                if v.notnan() {
                    n += 1;
                    for j in 0..i {
                        let a = *self.uget(j);
                        if a < v {
                            rank += 1.
                        } else if a == v {
                            n_repeat += 1
                        }
                    }
                } else {
                    rank = f64::NAN
                };
                let out = out.uget_mut(i);
                if n >= min_periods {
                    out.write(rank + 0.5 * (n_repeat - 1).f64())
                } else {
                    out.write(f64::NAN)
                };
            }
        }
        for (start, end) in (window - 1..self.len()).enumerate() {
            // 安全性：start和end不会超过self的长度，而out和self长度相同
            unsafe {
                let v = *self.uget(end);
                let mut n_repeat = 1; // 当前值的重复次数
                let mut rank = 1.; // 先假设为第一名，每当有元素比他更小，排名就加1
                if v.notnan() {
                    n += 1;
                    for i in start..end {
                        let a = *self.uget(i);
                        if a < v {
                            rank += 1.
                        } else if a == v {
                            n_repeat += 1
                        }
                    }
                } else {
                    rank = f64::NAN
                };
                let out = out.uget_mut(end);
                if n >= min_periods {
                    out.write(rank + 0.5 * (n_repeat - 1).f64()) // 对于重复值的method: average
                } else {
                    out.write(f64::NAN)
                };
                let v = *self.uget(start);
                if v.notnan() {
                    n -= 1;
                };
            }
        }
    }
);

impl_map_nd!(
    ts_rank_pct,
    pub fn ts_rank_pct_1d<S2>(
        &self,
        out: &mut ArrBase<S2, D>,
        window: usize,
        min_periods: usize,
    ) -> f64
    {where T: Number,}
    {
        let window = min(self.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {v.write(f64::NAN);});
        }
        let mut n = 0usize;
        for i in 0..window - 1 {
            // 安全性：i不会超过arr和out的长度
            unsafe {
                let v = *self.uget(i);
                let mut n_repeat = 1; // 当前值的重复次数
                let mut rank = 1.; // 先假设为第一名，每当有元素比他更小，排名就加1
                if v.notnan() {
                    n += 1;
                    for j in 0..i {
                        let a = *self.uget(j);
                        if a < v {
                            rank += 1.
                        } else if a == v {
                            n_repeat += 1
                        }
                    }
                } else {
                    rank = f64::NAN
                };
                let out = out.uget_mut(i);
                if n >= min_periods {
                    out.write((rank + 0.5 * (n_repeat - 1).f64()) / n.f64());
                } else {
                    out.write(f64::NAN);
                };
            }
        }
        for (start, end) in (window - 1..self.len()).enumerate() {
            // 安全性：start和end不会超过self的长度，而out和self长度相同
            unsafe {
                let v = *self.uget(end);
                let mut n_repeat = 1; // 当前值的重复次数
                let mut rank = 1.; // 先假设为第一名，每当有元素比他更小，排名就加1
                if v.notnan() {
                    n += 1;
                    for i in start..end {
                        let a = *self.uget(i);
                        if a < v {
                            rank += 1.
                        } else if a == v {
                            n_repeat += 1
                        }
                    }
                } else {
                    rank = f64::NAN
                };
                let out = out.uget_mut(end);
                if n >= min_periods {
                    out.write((rank + 0.5 * (n_repeat - 1).f64()) / n.f64()); // 对于重复值的method: average
                } else {
                    out.write(f64::NAN);
                };
                let v = *self.uget(start);
                if v.notnan() {
                    n -= 1;
                };
            }
        }
    }
);
// }
