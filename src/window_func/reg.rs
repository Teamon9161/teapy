use super::prelude::*;

pub fn ts_reg_1d<T: Number> (arr: ArrayView1<T>, mut out: ArrayViewMut1<f64>, window: usize, min_periods: usize, _out_step: usize)
where usize: AsPrimitive<T>
{   // 错位相减核心公式：sum_xt(t) = sum_xt(t-1) - sum_x + n * new_element
    if window == 1 {out.mapv_inplace(|_| f64::NAN); return;} // 如果滚动窗口是1则返回全nan
    assert!(window >= min_periods, "滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(out.len() == arr.len(), "输入数组的维度必须等于输出数组的维度");
    
    let mut sum = 0f64;
    let mut sum_xt = 0f64;
    let mut valid_window = 0usize;
    
    for i in 0..window-1 { // 安全性：i不会超过arr和out的长度
        let v = unsafe {(*arr.uget(i)).f64()};
        if v.notnan() {
            valid_window += 1; 
            sum_xt += valid_window.f64() * v; // 错位相减法, 忽略nan带来的系数和window不一致问题
            sum += v; 
        };
        unsafe {
            *out.uget_mut(i) = if valid_window >= min_periods {
                let w_64 = valid_window.f64();
                let nn_mul_n = valid_window * (valid_window + 1);
                let sum_t =  (nn_mul_n >> 1).f64(); // sum of time from 1 to window
                // denominator of slope
                let divisor = (valid_window * (nn_mul_n * (2 * valid_window + 1))).f64() / 6. - sum_t.powi(2);
                let slope = (w_64 * sum_xt - sum_t * sum) / divisor;
                let intercept = (sum - slope * sum_t) / w_64;
                intercept + slope * w_64
            } else { f64::NAN };
        }
    }
    let mut start = 0;
    for end in window-1..arr.len() {   
        // 安全性：start和end不会超过arr和out的长度
        unsafe {
            let v = (*arr.uget(end)).f64();
            if v.notnan() {
                valid_window += 1; 
                sum_xt += valid_window.f64() * v; // 错位相减法, 忽略nan带来的系数和window不一致问题
                sum += v; 
            };
            *out.uget_mut(end) = if valid_window >= min_periods {
                let w_64 = valid_window.f64();
                let nn_mul_n = valid_window * (valid_window + 1);
                let sum_t =  (nn_mul_n >> 1).f64(); // sum of time from 1 to window
                // denominator of slope
                let divisor = (valid_window * (nn_mul_n * (2 * valid_window + 1))).f64() / 6. - sum_t.powi(2);
                let slope = (w_64 * sum_xt - sum_t * sum) / divisor;
                let intercept = (sum - slope * sum_t) / w_64;
                intercept + slope * w_64
            } else { f64::NAN };
            let v = (*arr.uget(start)).f64();
            if v.notnan() {
                valid_window -= 1;
                sum_xt -= sum;
                sum -= v; 
            };
            start += 1;
        }
    }
}

pub fn ts_tsf_1d<T: Number> (arr: ArrayView1<T>, mut out: ArrayViewMut1<f64>, window: usize, min_periods: usize, _out_step: usize)
where usize: AsPrimitive<T>
// 固定window版本，速度较快，但是有nan的话会不准
{   
    if window == 1 {out.mapv_inplace(|_| f64::NAN); return;} // 如果滚动窗口是1则返回全nan
    assert!(window >= min_periods, "滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(out.len() == arr.len(), "输入数组的维度必须等于输出数组的维度");
    
    let mut sum = 0f64;
    let mut sum_xt = 0f64;
    let mut valid_window = 0usize;
    
    for i in 0..window-1 { // 安全性：i不会超过arr和out的长度
        let v = unsafe {(*arr.uget(i)).f64()};
        if v.notnan() {
            valid_window += 1; 
            sum_xt += valid_window.f64() * v; // 错位相减法, 忽略nan带来的系数和window不一致问题
            sum += v; 
        };
        unsafe {
            *out.uget_mut(i) = if valid_window >= min_periods {
                let w_64 = valid_window.f64();
                let nn_mul_n = valid_window * (valid_window + 1);
                let sum_t =  (nn_mul_n >> 1).f64() ; // sum of time from 1 to window
                // denominator of slope
                let divisor = (valid_window * (nn_mul_n * (2 * valid_window + 1))).f64() / 6. - sum_t.powi(2);
                let slope = (w_64 * sum_xt - sum_t * sum) / divisor;
                let intercept = (sum - slope * sum_t) / w_64;
                intercept + slope * (w_64 + 1.)
            } else { f64::NAN };
        }
    }
    let mut start = 0;
    for end in window-1..arr.len() {   
        // 安全性：start和end不会超过arr和out的长度
        unsafe {
            let v = (*arr.uget(end)).f64();
            if v.notnan() {
                valid_window += 1; 
                sum_xt += valid_window.f64() * v; // 错位相减法, 忽略nan带来的系数和window不一致问题
                sum += v; 
            };
            *out.uget_mut(end) = if valid_window >= min_periods {
                let w_64 = valid_window.f64();
                let nn_mul_n = valid_window * (valid_window + 1);
                let sum_t =  (nn_mul_n >> 1).f64() ; // sum of time from 1 to window
                // denominator of slope
                let divisor = (valid_window * (nn_mul_n * (2 * valid_window + 1))).f64() / 6. - sum_t.powi(2);
                let slope = (w_64 * sum_xt - sum_t * sum) / divisor;
                let intercept = (sum - slope * sum_t) / w_64;
                intercept + slope * (w_64 + 1.)
            } else { f64::NAN };
            let v = (*arr.uget(start)).f64();
            if v.notnan() {
                valid_window -= 1;
                sum_xt -= sum;
                sum -= v; 
            };
            start += 1;
        }
    }
}

pub fn ts_reg_slope_1d<T: Number> (arr: ArrayView1<T>, mut out: ArrayViewMut1<f64>, window: usize, min_periods: usize, _out_step: usize)
where usize: AsPrimitive<T>
// 固定window版本，速度较快，但是有nan的话会不准
{   
    if window == 1 {out.mapv_inplace(|_| f64::NAN); return;} // 如果滚动窗口是1则返回全nan
    assert!(window >= min_periods, "滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(out.len() == arr.len(), "输入数组的维度必须等于输出数组的维度");
    
    let mut sum = 0f64;
    let mut sum_xt = 0f64;
    let mut valid_window = 0usize;
    
    for i in 0..window-1 { // 安全性：i不会超过arr和out的长度
        let v = unsafe {(*arr.uget(i)).f64()};
        if v.notnan() {
            valid_window += 1; 
            sum_xt += valid_window.f64() * v; // 错位相减法, 忽略nan带来的系数和window不一致问题
            sum += v; 
        };
        unsafe {
            *out.uget_mut(i) = if valid_window >= min_periods {
                let w_64 = valid_window.f64();
                let nn_mul_n = valid_window * (valid_window + 1);
                let sum_t =  (nn_mul_n >> 1).f64() ; // sum of time from 1 to window
                // denominator of slope
                let divisor = (valid_window * (nn_mul_n * (2 * valid_window + 1))).f64() / 6. - sum_t.powi(2);
                let slope = (w_64 * sum_xt - sum_t * sum) / divisor;
                slope
            } else { f64::NAN };
        }
    }
    let mut start = 0;
    for end in window-1..arr.len() {   
        // 安全性：start和end不会超过arr和out的长度
        unsafe {
            let v = (*arr.uget(end)).f64();
            if v.notnan() {
                valid_window += 1; 
                sum_xt += valid_window.f64() * v; // 错位相减法, 忽略nan带来的系数和window不一致问题
                sum += v; 
            };
            *out.uget_mut(end) = if valid_window >= min_periods {
                let w_64 = valid_window.f64();
                let nn_mul_n = valid_window * (valid_window + 1);
                let sum_t =  (nn_mul_n >> 1).f64() ; // sum of time from 1 to window
                // denominator of slope
                let divisor = (valid_window * (nn_mul_n * (2 * valid_window + 1))).f64() / 6. - sum_t.powi(2);
                let slope = (w_64 * sum_xt - sum_t * sum) / divisor;
                slope
            } else { f64::NAN };
            let v = (*arr.uget(start)).f64();
            if v.notnan() {
                valid_window -= 1;
                sum_xt -= sum;
                sum -= v; 
            };
            start += 1;
        }
    }
}


pub fn ts_reg_intercept_1d<T: Number> (arr: ArrayView1<T>, mut out: ArrayViewMut1<f64>, window: usize, min_periods: usize, _out_step: usize)
where usize: AsPrimitive<T>
// 固定window版本，速度较快，但是有nan的话会不准
{   
    if window == 1 {out.mapv_inplace(|_| f64::NAN); return;} // 如果滚动窗口是1则返回全nan
    assert!(window >= min_periods, "滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(out.len() == arr.len(), "输入数组的维度必须等于输出数组的维度");
    
    let mut sum = 0f64;
    let mut sum_xt = 0f64;
    let mut valid_window = 0usize;
    
    for i in 0..window-1 { // 安全性：i不会超过arr和out的长度
        let v = unsafe {(*arr.uget(i)).f64()};
        if v.notnan() {
            valid_window += 1; 
            sum_xt += valid_window.f64() * v; // 错位相减法, 忽略nan带来的系数和window不一致问题
            sum += v; 
        };
        unsafe {
            *out.uget_mut(i) = if valid_window >= min_periods {
                let w_64 = valid_window.f64();
                let nn_mul_n = valid_window * (valid_window + 1);
                let sum_t =  (nn_mul_n >> 1).f64() ; // sum of time from 1 to window
                // denominator of slope
                let divisor = (valid_window * (nn_mul_n * (2 * valid_window + 1))).f64() / 6. - sum_t.powi(2);
                let slope = (w_64 * sum_xt - sum_t * sum) / divisor;
                let intercept = (sum - slope * sum_t) / w_64;
                intercept
            } else { f64::NAN };
        }
    }
    let mut start = 0;
    for end in window-1..arr.len() {   
        // 安全性：start和end不会超过arr和out的长度
        unsafe {
            let v = (*arr.uget(end)).f64();
            if v.notnan() {
                valid_window += 1; 
                sum_xt += valid_window.f64() * v; // 错位相减法, 忽略nan带来的系数和window不一致问题
                sum += v; 
            };
            *out.uget_mut(end) = if valid_window >= min_periods {
                let w_64 = valid_window.f64();
                let nn_mul_n = valid_window * (valid_window + 1);
                let sum_t =  (nn_mul_n >> 1).f64() ; // sum of time from 1 to window
                // denominator of slope
                let divisor = (valid_window * (nn_mul_n * (2 * valid_window + 1))).f64() / 6. - sum_t.powi(2);
                let slope = (w_64 * sum_xt - sum_t * sum) / divisor;
                let intercept = (sum - slope * sum_t) / w_64;
                intercept
            } else { f64::NAN };
            let v = (*arr.uget(start)).f64();
            if v.notnan() {
                valid_window -= 1;
                sum_xt -= sum;
                sum -= v; 
            };
            start += 1;
        }
    }
}
