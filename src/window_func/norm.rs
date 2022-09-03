use super::prelude::*;

pub fn ts_stable<T: Number>(
    arr: ArrayView1<T>,
    mut out: ArrayViewMut1<f64>,
    window: usize,
    min_periods: usize,
    _out_step: usize,
) where
    usize: AsPrimitive<T>,
{
    if window == 1 {
        out.mapv_inplace(|_| f64::NAN);
        return;
    } // 如果滚动窗口是1则返回全nan
    let min_periods = min_periods.max(2);
    assert!(window >= min_periods, "滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(
        out.len() == arr.len(),
        "输入数组的维度必须等于输出数组的维度"
    );

    let mut sum = 0f64;
    let mut sum2 = 0f64;
    let mut valid_window = 0usize;
    for i in 0..window - 1 {
        // 安全性：i不会超过arr和out的长度
        let v = unsafe { (*arr.uget(i)).f64() };
        if v.notnan() {
            valid_window += 1;
            sum += v;
            sum2 += v * v
        };
        unsafe {
            *out.uget_mut(i) = if valid_window >= min_periods {
                let v_window = valid_window.f64();
                let mut var = sum2 / v_window;
                let mean = sum / v_window;
                var -= mean.powi(2);
                if var > 1e-14 {
                    mean / (var * v_window / (v_window - 1f64)).sqrt()
                } else {
                    f64::NAN
                }
            } else {
                f64::NAN
            };
        }
    }
    for (start, end) in (window - 1..arr.len()).enumerate() {
        // 安全性：start和end不会超过arr和out的长度
        unsafe {
            let v = (*arr.uget(end)).f64();
            if v.notnan() {
                valid_window += 1;
                sum += v;
                sum2 += v * v
            };
            *out.uget_mut(end) = if valid_window >= min_periods {
                let v_window = valid_window.f64();
                let mut var = sum2 / v_window;
                let mean = sum / v_window;
                var -= mean.powi(2);
                if var > 1e-14 {
                    mean / (var * v_window / (v_window - 1f64)).sqrt()
                } else {
                    f64::NAN
                }
            } else {
                f64::NAN
            };
            let v = (*arr.uget(start)).f64();
            if v.notnan() {
                valid_window -= 1;
                sum -= v;
                sum2 -= v * v
            };
        }
    }
}

pub fn ts_meanstdnorm<T: Number>(
    arr: ArrayView1<T>,
    mut out: ArrayViewMut1<f64>,
    window: usize,
    min_periods: usize,
    _out_step: usize,
) where
    usize: AsPrimitive<T>,
{
    if window == 1 {
        out.mapv_inplace(|_| f64::NAN);
        return;
    } // 如果滚动窗口是1则返回全nan
    let min_periods = min_periods.max(2);
    assert!(window >= min_periods, "滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(
        out.len() == arr.len(),
        "输入数组的维度必须等于输出数组的维度"
    );

    let mut sum = 0f64;
    let mut sum2 = 0f64;
    let mut valid_window = 0usize;
    for i in 0..window - 1 {
        // 安全性：i不会超过arr和out的长度
        let v = unsafe { (*arr.uget(i)).f64() };
        if v.notnan() {
            valid_window += 1;
            sum += v;
            sum2 += v * v
        };
        unsafe {
            *out.uget_mut(i) = if valid_window >= min_periods {
                let v_window = valid_window.f64();
                let mut var = sum2 / v_window;
                let mean = sum / v_window;
                var -= mean.powi(2);
                if var > 1e-14 {
                    (v - mean) / (var * v_window / (v_window - 1.)).sqrt()
                } else {
                    f64::NAN
                }
            } else {
                f64::NAN
            };
        }
    }
    for (start, end) in (window - 1..arr.len()).enumerate() {
        // 安全性：start和end不会超过arr和out的长度
        unsafe {
            let v = (*arr.uget(end)).f64();
            if v.notnan() {
                valid_window += 1;
                sum += v;
                sum2 += v * v
            };
            *out.uget_mut(end) = if valid_window >= min_periods {
                let v_window = valid_window.f64();
                let mut var = sum2 / v_window;
                let mean = sum / v_window;
                var -= mean.powi(2);
                if var > 1e-14 {
                    (v - mean) / (var * v_window / (v_window - 1.)).sqrt()
                } else {
                    f64::NAN
                }
            } else {
                f64::NAN
            };
            let v = (*arr.uget(start)).f64();
            if v.notnan() {
                valid_window -= 1;
                sum -= v;
                sum2 -= v * v
            };
        }
    }
}

pub fn ts_minmaxnorm<T: Number>(
    arr: ArrayView1<T>,
    mut out: ArrayViewMut1<f64>,
    window: usize,
    min_periods: usize,
    _out_step: usize,
) where
    usize: AsPrimitive<T>,
{
    if window == 1 {
        out.mapv_inplace(|_| f64::NAN);
        return;
    } // 如果滚动窗口是1则返回全nan
    assert!(window >= min_periods, "滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(
        out.len() == arr.len(),
        "输入数组的维度必须等于输出数组的维度"
    );
    let mut max: T = T::min_();
    let mut max_idx = 0usize;
    let mut min: T = T::max_();
    let mut min_idx = 0usize;
    let mut valid_window = 0usize;
    for i in 0..window - 1 {
        // 安全性：i不会超过arr和out的长度
        let v = unsafe { *arr.uget(i) };
        if v.notnan() {
            valid_window += 1
        };
        if v >= max {
            (max, max_idx) = (v, i);
        }
        if v <= min {
            (min, min_idx) = (v, i);
        }
        unsafe {
            *out.uget_mut(i) = if (valid_window >= min_periods) & (max != min) {
                (v - min).f64() / (max - min).f64()
            } else {
                f64::NAN
            };
        }
    }
    for (start, end) in (window - 1..arr.len()).enumerate() {
        // 安全性：start和end不会超过arr和out的长度
        unsafe {
            let v = *arr.uget(end);
            if v.notnan() {
                valid_window += 1
            };
            match (max_idx < start, min_idx < start) {
                (true, false) => {
                    // 最大值已经失效，重新找最大值
                    max = T::min_();
                    for i in start..end {
                        let v = *arr.uget(i);
                        if v >= max {
                            (max, max_idx) = (v, i);
                        }
                    }
                }
                (false, true) => {
                    // 最小值已经失效，重新找最小值
                    min = T::max_();
                    for i in start..end {
                        let v = *arr.uget(i);
                        if v <= min {
                            (min, min_idx) = (v, i);
                        }
                    }
                }
                (true, true) => {
                    // 最大和最小值都已经失效，重新找最大和最小值
                    (max, min) = (T::min_(), T::max_());
                    for i in start..end {
                        let v = *arr.uget(i);
                        if v >= max {
                            (max, max_idx) = (v, i);
                        }
                        if v <= min {
                            (min, min_idx) = (v, i);
                        }
                    }
                }
                (false, false) => (), // 不需要重新找最大和最小值
            }
            // 检查end位置是否是最大或最小值
            if v >= max {
                (max, max_idx) = (v, end);
            }
            if v <= min {
                (min, min_idx) = (v, end);
            }

            *out.uget_mut(end) = if (valid_window >= min_periods) & (max != min) {
                (v - min).f64() / (max - min).f64()
            } else {
                f64::NAN
            };
            if NotNan!(*arr.uget(start)) {
                valid_window -= 1
            };
        }
    }
}
