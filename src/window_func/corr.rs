use super::prelude::*;

pub fn ts_cov_1d<T: Number, U: Number>(
    arr1: ArrayView1<T>,
    arr2: ArrayView1<U>,
    mut out: ArrayViewMut1<f64>,
    window: usize,
    min_periods: usize,
    _out_step: usize,
) where
    usize: Number,
{
    if window == 1 {
        return;
    } // out默认是全0数列, 如果滚动窗口是1则返回全0
    let min_periods = min_periods.max(2);
    assert!(window >= min_periods, "滚动window小于最小期数");
    assert!(arr1.len() == arr2.len(), "ts_cov中两个array长度不相等");
    assert!(window <= arr1.len(), "滚动window大于数组长度");
    debug_assert!(
        out.len() == arr1.len(),
        "输入数组的维度必须等于输出数组的维度"
    );

    let mut sum_a = 0f64;
    let mut sum_b = 0f64;
    let mut sum_ab = 0f64;
    let mut valid_window = 0usize;
    for i in 0..window - 1 {
        // 安全性：i不会超过arr和out的长度
        let (v_a, v_b) = unsafe { ((*arr1.uget(i)).f64(), (*arr2.uget(i)).f64()) };
        if v_a.notnan() & v_b.notnan() {
            valid_window += 1;
            sum_a += v_a;
            sum_b += v_b;
            sum_ab += v_a * v_b;
        };
        unsafe {
            *out.uget_mut(i) = if valid_window >= min_periods {
                let v_window = valid_window.f64();
                let e_xy = sum_ab / v_window;
                let exey = (sum_a * sum_b) / v_window.powi(2);
                (e_xy - exey) * v_window / (v_window - 1f64)
            } else {
                f64::NAN
            };
        }
    }
    for (start, end) in (window - 1..arr1.len()).enumerate() {
        // 安全性：start和end不会超过arr和out的长度
        unsafe {
            let (v_a, v_b) = ((*arr1.uget(end)).f64(), (*arr2.uget(end)).f64());
            if v_a.notnan() & v_b.notnan() {
                valid_window += 1;
                sum_a += v_a;
                sum_b += v_b;
                sum_ab += v_a * v_b;
            };
            *out.uget_mut(end) = if valid_window >= min_periods {
                let v_window = valid_window.f64();
                let e_xy = sum_ab / v_window;
                let exey = (sum_a * sum_b) / v_window.powi(2);
                (e_xy - exey) * v_window / (v_window - 1f64)
            } else {
                f64::NAN
            };
            let (v_a, v_b) = ((*arr1.uget(start)).f64(), (*arr2.uget(start)).f64());
            if v_a.notnan() & v_b.notnan() {
                valid_window -= 1;
                sum_a -= v_a;
                sum_b -= v_b;
                sum_ab -= v_a * v_b;
            };
        }
    }
}

pub fn ts_corr_1d<T: Number, U: Number>(
    arr1: ArrayView1<T>,
    arr2: ArrayView1<U>,
    mut out: ArrayViewMut1<f64>,
    window: usize,
    min_periods: usize,
    _out_step: usize,
) where
    usize: Number,
{
    if window == 1 {
        return;
    } // out默认是全0数列, 如果滚动窗口是1则返回全0
    let min_periods = min_periods.max(2);
    assert!(window >= min_periods, "滚动window小于最小期数");
    assert!(arr1.len() == arr2.len(), "ts_corr中两个array长度不相等");
    assert!(window <= arr1.len(), "滚动window大于数组长度");
    debug_assert!(
        out.len() == arr1.len(),
        "输入数组的维度必须等于输出数组的维度"
    );

    let mut sum_a = 0f64;
    let mut sum2_a = 0f64;
    let mut sum_b = 0f64;
    let mut sum2_b = 0f64;
    let mut sum_ab = 0f64;
    let mut valid_window = 0usize;
    for i in 0..window - 1 {
        // 安全性：i不会超过arr和out的长度
        let (v_a, v_b) = unsafe { ((*arr1.uget(i)).f64(), (*arr2.uget(i)).f64()) };
        if v_a.notnan() & v_b.notnan() {
            valid_window += 1;
            sum_a += v_a;
            sum2_a += v_a * v_a;
            sum_b += v_b;
            sum2_b += v_b * v_b;
            sum_ab += v_a * v_b;
        };
        unsafe {
            *out.uget_mut(i) = if valid_window >= min_periods {
                let v_window = valid_window.f64();
                let mean_a = sum_a / v_window;
                let mut var_a = sum2_a / v_window;
                let mean_b = sum_b / v_window;
                let mut var_b = sum2_b / v_window;
                let exy = sum_ab / v_window;
                let exey = sum_a * sum_b / v_window.powi(2);
                var_a -= mean_a.powi(2);
                var_b -= mean_b.powi(2);
                if (var_a > 1e-14) & (var_b > 1e-14) {
                    (exy - exey) / (var_a * var_b).sqrt()
                } else {
                    f64::NAN
                }
            } else {
                f64::NAN
            };
        }
    }
    for (start, end) in (window - 1..arr1.len()).enumerate() {
        // 安全性：start和end不会超过arr和out的长度
        unsafe {
            let (v_a, v_b) = ((*arr1.uget(end)).f64(), (*arr2.uget(end)).f64());
            if v_a.notnan() & v_b.notnan() {
                valid_window += 1;
                sum_a += v_a;
                sum2_a += v_a * v_a;
                sum_b += v_b;
                sum2_b += v_b * v_b;
                sum_ab += v_a * v_b;
            };
            *out.uget_mut(end) = if valid_window >= min_periods {
                let v_window = valid_window.f64();
                let mean_a = sum_a / v_window;
                let mut var_a = sum2_a / v_window;
                let mean_b = sum_b / v_window;
                let mut var_b = sum2_b / v_window;
                let exy = sum_ab / v_window;
                let exey = sum_a * sum_b / v_window.powi(2);
                var_a -= mean_a.powi(2);
                var_b -= mean_b.powi(2);
                if (var_a > 1e-14) & (var_b > 1e-14) {
                    (exy - exey) / (var_a * var_b).sqrt()
                } else {
                    f64::NAN
                }
            } else {
                f64::NAN
            };
            let (v_a, v_b) = ((*arr1.uget(start)).f64(), (*arr2.uget(start)).f64());
            if v_a.notnan() & v_b.notnan() {
                valid_window -= 1;
                sum_a -= v_a;
                sum2_a -= v_a * v_a;
                sum_b -= v_b;
                sum2_b -= v_b * v_b;
                sum_ab -= v_a * v_b;
            };
        }
    }
}
