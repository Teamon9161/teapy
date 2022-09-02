use super::prelude::*;

pub fn ts_sma_1d<T: Number>(
    arr: ArrayView1<T>,
    mut out: ArrayViewMut1<f64>,
    window: usize,
    min_periods: usize,
    _out_step: usize,
) where
    usize: AsPrimitive<T>,
{
    // if let DataType::F64T = T::dtype() {
    //     unsafe { // 直接调用c++代码
    //         ffi::ts_sma_1d(
    //             arr.as_ptr() as *const f64,
    //             out.as_mut_ptr() as *mut f64,
    //             arr.len() as i32,
    //             window as i32,
    //             min_periods as i32,
    //             out_step as i32,
    //         )
    //     }
    // }
    // else {

    assert!(window >= min_periods, "滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(
        out.len() == arr.len(),
        "输入数组的维度必须等于输出数组的维度"
    );
    let mut sum = 0f64;
    let mut valid_window = 0usize;
    for i in 0..window - 1 {
        // 安全性：i不会超过arr和out的长度
        unsafe {
            let v = (*arr.uget(i)).f64();
            if v.notnan() {
                sum += v;
                valid_window += 1
            };
            *out.uget_mut(i) = if valid_window >= min_periods {
                sum / valid_window.f64()
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
                sum += v;
                valid_window += 1
            };
            *out.uget_mut(end) = if valid_window >= min_periods {
                sum.f64() / valid_window.f64()
            } else {
                f64::NAN
            };
            let v = (*arr.uget(start)).f64();
            if v.notnan() {
                sum -= v;
                valid_window -= 1
            };
        }
    }
}

pub fn ts_ewm_1d<T: Number>(
    arr: ArrayView1<T>,
    mut out: ArrayViewMut1<f64>,
    window: usize,
    min_periods: usize,
    _out_step: usize,
) where
    usize: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    // 错位相减核心公式：
    // q_x(t) = 1 * new_element - alpha(q_x(t-1 without 1st element)) - 1st element * oma ^ (n-1)
    assert!(window >= min_periods, "滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(
        out.len() == arr.len(),
        "输入数组的维度必须等于输出数组的维度"
    );
    let mut q_x = 0.; // 权重的分子部分 * 元素，使用错位相减法来计算
    let mut valid_window = 0usize;
    let alpha = 2. / window.f64();
    let oma = 1. - alpha; // one minus alpha
    for i in 0..window - 1 {
        // 安全性：i不会超过arr和out的长度
        unsafe {
            let v = *arr.uget(i);
            if v.notnan() {
                valid_window += 1;
                q_x += v.f64() - alpha * q_x.f64();
            };
            *out.uget_mut(i) = if valid_window >= min_periods {
                q_x.f64() * alpha / (1. - oma.powi(valid_window as i32))
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
                valid_window += 1;
                q_x += v.f64() - alpha * q_x.f64();
            };
            *out.uget_mut(end) = if valid_window >= min_periods {
                q_x.f64() * alpha / (1. - oma.powi(valid_window as i32))
            } else {
                f64::NAN
            };
            let v = *arr.uget(start);
            if v.notnan() {
                valid_window -= 1;
                q_x -= v.f64() * oma.powi(valid_window as i32); // 本应是window-1，不过本身window就要自然减一，调整一下顺序
            };
        }
    }
}

pub fn ts_wma_1d<T: Number>(
    arr: ArrayView1<T>,
    mut out: ArrayViewMut1<f64>,
    window: usize,
    min_periods: usize,
    _out_step: usize,
) where
    usize: AsPrimitive<T>,
{
    // 错位相减核心公式：sum_xt(t) = sum_xt(t-1) - sum_x + n * new_element
    assert!(window >= min_periods, "滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(
        out.len() == arr.len(),
        "输入数组的维度必须等于输出数组的维度"
    );

    let mut sum = 0f64;
    let mut sum_xt = 0f64;
    let mut valid_window = 0;

    for i in 0..window - 1 {
        // 安全性：i不会超过arr和out的长度
        let v = unsafe { (*arr.uget(i)).f64() };
        if v.notnan() {
            valid_window += 1;
            sum_xt += valid_window.f64() * v; // 错位相减法, 忽略nan带来的系数和window不一致问题
            sum += v;
        };
        unsafe {
            *out.uget_mut(i) = if valid_window >= min_periods {
                let divisor = (valid_window * (valid_window + 1)) >> 1;
                sum_xt / divisor.f64()
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
                sum_xt += valid_window.f64() * v; // 错位相减法, 忽略nan带来的系数和window不一致问题
                sum += v;
            };
            *out.uget_mut(end) = if valid_window >= min_periods {
                let divisor = (valid_window * (valid_window + 1)) >> 1;
                sum_xt / divisor.f64()
            } else {
                f64::NAN
            };
            let v = (*arr.uget(start)).f64();
            if v.notnan() {
                valid_window -= 1;
                sum_xt -= sum;
                sum -= v;
            };
        }
    }
}

pub fn ts_sum_1d<T: Number>(
    arr: ArrayView1<T>,
    mut out: ArrayViewMut1<f64>,
    window: usize,
    min_periods: usize,
    _out_step: usize,
) where
    usize: AsPrimitive<T>,
{
    assert!(window >= min_periods, "滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(
        out.len() == arr.len(),
        "输入数组的维度必须等于输出数组的维度"
    );
    let mut sum = 0f64;
    let mut valid_window = 0usize;
    for i in 0..window - 1 {
        // 安全性：i不会超过arr和out的长度
        unsafe {
            let v = (*arr.uget(i)).f64();
            if v.notnan() {
                sum += v;
                valid_window += 1
            };
            *out.uget_mut(i) = if valid_window >= min_periods {
                sum
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
                sum += v;
                valid_window += 1
            };
            *out.uget_mut(end) = if valid_window >= min_periods {
                sum
            } else {
                f64::NAN
            };
            let v = (*arr.uget(start)).f64();
            if v.notnan() {
                sum -= v;
                valid_window -= 1
            };
        }
    }
}

pub fn ts_std_1d<T: Number>(
    arr: ArrayView1<T>,
    mut out: ArrayViewMut1<f64>,
    window: usize,
    min_periods: usize,
    _out_step: usize,
) where
    usize: AsPrimitive<T>,
{
    if window == 1 {
        return;
    } // out默认是全0数列, 如果滚动窗口是1则返回全0
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
                // var肯定大于等于0，否则只能是精度问题
                if var > 0. {
                    (var * v_window / (v_window - 1f64)).sqrt()
                } else {
                    0.
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
                // var肯定大于等于0，否则只能是精度问题
                if var > 0. {
                    (var * v_window / (v_window - 1f64)).sqrt()
                } else {
                    0.
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

pub fn ts_skew_1d<T: Number>(
    arr: ArrayView1<T>,
    mut out: ArrayViewMut1<f64>,
    window: usize,
    min_periods: usize,
    _out_step: usize,
) where
    usize: AsPrimitive<T>,
{
    if window == 1 {
        return;
    }
    // out默认是全0数列, 如果滚动窗口是1则返回全0
    else if window == 2 {
        out.mapv_inplace(|_| f64::NAN);
        return;
    } // 如果滚动窗口是2则返回全nan
    let min_periods = min_periods.max(3);
    assert!(window >= min_periods, "ts_skew滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(
        out.len() == arr.len(),
        "输入数组的维度必须等于输出数组的维度"
    );
    let mut sum = 0f64;
    let mut sum2 = 0f64;
    let mut sum3 = 0f64;
    let mut valid_window = 0usize;
    for i in 0..window - 1 {
        // 安全性：i不会超过arr和out的长度
        let v = unsafe { (*arr.uget(i)).f64() };
        if v.notnan() {
            valid_window += 1;
            sum += v;
            let v2 = v * v;
            sum2 += v2;
            sum3 += v2 * v;
        };
        unsafe {
            *out.uget_mut(i) = if valid_window >= min_periods {
                let v_window = valid_window.f64();
                let mut var = sum2 / v_window;
                let mut mean = sum / v_window; // mean
                var -= mean.powi(2); // var
                if var <= 0. {
                    0.
                }
                // 标准差为0， 则偏度为0
                else {
                    let std = var.sqrt(); // std
                    let res = sum3 / v_window; // Ex^3
                    mean /= std; // mean / std
                    let adjust =
                        (valid_window * (valid_window - 1)).f64().sqrt() / (valid_window - 2).f64();
                    adjust * (res / std.powi(3) - 3_f64 * mean - mean.powi(3))
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
                let v2 = v * v;
                sum2 += v2;
                sum3 += v2 * v;
            };
            *out.uget_mut(end) = if valid_window >= min_periods {
                let v_window = valid_window.f64();
                let mut var = sum2 / v_window;
                let mut mean = sum / v_window; // mean
                var -= mean.powi(2); // var
                if var <= 0. {
                    0.
                }
                // 标准差为0， 则偏度为0
                else {
                    let std = var.sqrt(); // std
                    let res = sum3 / v_window; // Ex^3
                    mean /= std; // mean / std
                    let adjust =
                        (valid_window * (valid_window - 1)).f64().sqrt() / (valid_window - 2).f64();
                    adjust * (res / std.powi(3) - 3_f64 * mean - mean.powi(3))
                }
            } else {
                f64::NAN
            };
            let v = (*arr.uget(start)).f64();
            if v.notnan() {
                valid_window -= 1;
                sum -= v;
                let v2 = v * v;
                sum2 -= v2;
                sum3 -= v2 * v;
            };
        }
    }
}

pub fn ts_kurt_1d<T: Number>(
    arr: ArrayView1<T>,
    mut out: ArrayViewMut1<f64>,
    window: usize,
    min_periods: usize,
    _out_step: usize,
) where
    usize: AsPrimitive<T>,
{
    if window < 4 {
        out.mapv_inplace(|_| f64::NAN);
        return;
    } // 如果滚动窗口小于4则返回全nan
    let min_periods = min_periods.max(4);
    assert!(window >= min_periods, "ts_kurt滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(
        out.len() == arr.len(),
        "输入数组的维度必须等于输出数组的维度"
    );

    let mut sum = 0f64;
    let mut sum2 = 0f64;
    let mut sum3 = 0f64;
    let mut sum4 = 0f64;
    let mut valid_window = 0usize;
    for i in 0..window - 1 {
        // 安全性：i不会超过arr和out的长度
        let v = unsafe { (*arr.uget(i)).f64() };
        if v.notnan() {
            valid_window += 1;
            sum += v;
            let v2 = v * v;
            sum2 += v2;
            sum3 += v2 * v;
            sum4 += v2 * v2;
        };
        unsafe {
            *out.uget_mut(i) = if valid_window >= min_periods {
                let v_window = valid_window.f64();
                let mean = sum / v_window; // Ex
                let ex2 = sum2 / v_window; // Ex^2
                let var = ex2 - mean.powi(2); // var
                if var <= 0. {
                    0.
                }
                // 方差为0， 则峰度为0
                else {
                    let var2 = var * var; // var^2
                    let ex4 = sum4 / v_window; // Ex^4
                    let ex3 = sum3 / v_window; // Ex^3
                    let mean2_var = mean * mean / var; // (mean / std)^2
                    let out =
                        (ex4 - 4. * mean * ex3) / var2 + 6. * mean2_var + 3. * mean2_var.powi(2);
                    let n = v_window;
                    1. / ((n - 2.) * (n - 3.)) * ((n.powi(2) - 1.) * out - 3. * (n - 1.).powi(2))
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
                let v2 = v * v;
                sum2 += v2;
                sum3 += v2 * v;
                sum4 += v2 * v2;
            };
            *out.uget_mut(end) = if valid_window >= min_periods {
                let v_window = valid_window.f64();
                let mean = sum / v_window; // Ex
                let ex2 = sum2 / v_window; // Ex^2
                let var = ex2 - mean.powi(2); // var
                if var <= 0. {
                    0.
                }
                // 方差为0， 则峰度为0
                else {
                    let var2 = var * var; // var^2
                    let ex4 = sum4 / v_window; // Ex^4
                    let ex3 = sum3 / v_window; // Ex^3
                    let mean2_var = mean * mean / var; // (mean / std)^2
                    let out =
                        (ex4 - 4. * mean * ex3) / var2 + 6. * mean2_var + 3. * mean2_var.powi(2);
                    let n = v_window;
                    1. / ((n - 2.) * (n - 3.)) * ((n.powi(2) - 1.) * out - 3. * (n - 1.).powi(2))
                }
            } else {
                f64::NAN
            };
            let v = (*arr.uget(start)).f64();
            if v.notnan() {
                valid_window -= 1;
                sum -= v;
                let v2 = v * v;
                sum2 -= v2;
                sum3 -= v2 * v;
                sum4 -= v2 * v2;
            };
        }
    }
}

pub fn ts_prod_1d<T: Number>(
    arr: ArrayView1<T>,
    mut out: ArrayViewMut1<f64>,
    window: usize,
    min_periods: usize,
    _out_step: usize,
) where
    usize: AsPrimitive<T>,
{
    assert!(window >= min_periods, "滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(
        out.len() == arr.len(),
        "输入数组的维度必须等于输出数组的维度"
    );
    let mut prod = 1_f64;
    let mut valid_window = 0usize;
    let mut zero_num = 0usize;
    for i in 0..window - 1 {
        // 安全性：i不会超过arr和out的长度
        unsafe {
            let v = *arr.uget(i);
            if v.notnan() {
                if v != T::zero() {
                    prod *= v.f64();
                } else {
                    zero_num += 1;
                }
                valid_window += 1;
            };
            *out.uget_mut(i) = if valid_window >= min_periods {
                if zero_num == 0 {
                    prod
                } else {
                    0_f64
                }
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
                if v != T::zero() {
                    prod *= v.f64();
                } else {
                    zero_num += 1;
                }
                valid_window += 1;
            };
            *out.uget_mut(end) = if valid_window >= min_periods {
                if zero_num == 0 {
                    prod
                } else {
                    0_f64
                }
            } else {
                f64::NAN
            };
            let v = *arr.uget(start);
            if v.notnan() {
                valid_window -= 1;
                if v != T::zero() {
                    prod /= v.f64();
                } else {
                    zero_num -= 1;
                }
            };
        }
    }
}

pub fn ts_prod_mean_1d<T: Number>(
    arr: ArrayView1<T>,
    mut out: ArrayViewMut1<f64>,
    window: usize,
    min_periods: usize,
    _out_step: usize,
) where
    usize: AsPrimitive<T>,
{
    assert!(window >= min_periods, "滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(
        out.len() == arr.len(),
        "输入数组的维度必须等于输出数组的维度"
    );
    let mut prod = 1f64;
    let mut valid_window = 0usize;
    let mut zero_num = 0usize;
    for i in 0..window - 1 {
        // 安全性：i不会超过arr和out的长度
        unsafe {
            let v = *arr.uget(i);
            if v.notnan() {
                if v != T::zero() {
                    prod *= v.f64();
                } else {
                    zero_num += 1;
                }
                valid_window += 1;
            };
            *out.uget_mut(i) = if valid_window >= min_periods {
                if zero_num == 0 {
                    prod.powf(1. / valid_window.f64())
                } else {
                    0_f64
                }
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
                if v != T::zero() {
                    prod *= v.f64();
                } else {
                    zero_num += 1;
                }
                valid_window += 1;
            };
            *out.uget_mut(end) = if valid_window >= min_periods {
                if zero_num == 0 {
                    prod.powf(1. / valid_window.f64())
                } else {
                    0_f64
                }
            } else {
                f64::NAN
            };
            let v = *arr.uget(start);
            if v.notnan() {
                valid_window -= 1;
                if v != T::zero() {
                    prod /= v.f64();
                } else {
                    zero_num -= 1;
                }
            };
        }
    }
}
