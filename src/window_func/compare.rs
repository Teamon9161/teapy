use super::prelude::*;


pub fn ts_max_1d<T: Number> (arr: ArrayView1<T>, mut out: ArrayViewMut1<f64>, window: usize, min_periods: usize, _out_step: usize)
where usize: AsPrimitive<T>
{   
    assert!(window >= min_periods, "滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(out.len() == arr.len(), "输入数组的维度必须等于输出数组的维度");
    let mut max: T = T::min_();
    let mut max_idx = 0usize;
    let mut valid_window = 0usize;
    for i in 0..window-1 { // 安全性：i不会超过arr和out的长度
        let v = unsafe {*arr.uget(i)};
        if v.notnan() {valid_window += 1};
        if v > max {(max, max_idx) = (v, i);}
        unsafe {
            *out.uget_mut(i) = if valid_window >= min_periods {
                max.f64()
            } else { f64::NAN };
        }
    }
    let mut start = 0;
    for end in window-1..arr.len() {   
        // 安全性：start和end不会超过arr和out的长度
        unsafe {
            let v = *arr.uget(end);
            if v.notnan() {valid_window += 1};
            if max_idx < start { // 最大值已经失效，重新找到最大值
                let v = *arr.uget(start);
                max = if IsNan!(v) {T::min_()} else {v};
                for i in start..=end {
                    let v = *arr.uget(i);
                    if v >= max {(max, max_idx) = (v, i);}
                }
            } else if v >= max {  // 当前是最大值
                (max, max_idx) = (v, end);
            }
            *out.uget_mut(end) = if valid_window >= min_periods {
                max.f64()
            } else { f64::NAN };
            if NotNan!({*arr.uget(start)}) {valid_window -= 1};
            start += 1;
        }
    }
}

pub fn ts_argmax_1d<T: Number> (arr: ArrayView1<T>, mut out: ArrayViewMut1<f64>, window: usize, min_periods: usize, _out_step: usize)
where usize: AsPrimitive<T>
{   
    assert!(window >= min_periods, "滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(out.len() == arr.len(), "输入数组的维度必须等于输出数组的维度");
    let mut max: T = T::min_();
    let mut max_idx = 0usize;
    let mut valid_window = 0usize;
    for i in 0..window-1 { // 安全性：i不会超过arr和out的长度
        let v = unsafe {*arr.uget(i)};
        if v.notnan() {valid_window += 1};
        if v > max {(max, max_idx) = (v, i);}
        unsafe {
            *out.uget_mut(i) = if valid_window >= min_periods {
                (max_idx + 1).f64()
            } else { f64::NAN };
        }
    }
    let mut start = 0;
    for end in window-1..arr.len() {   
        // 安全性：start和end不会超过arr和out的长度
        unsafe {
            let v = *arr.uget(end);
            if v.notnan() {valid_window += 1};
            if max_idx < start { // 最大值已经失效，重新找到最大值
                let v = *arr.uget(start);
                max = if IsNan!(v) {T::min_()} else {v};
                for i in start..=end {
                    let v = *arr.uget(i);
                    if v >= max {(max, max_idx) = (v, i);}
                }
            } else if v >= max {  // 当前是最大值
                (max, max_idx) = (v, end);
            }
            *out.uget_mut(end) = if valid_window >= min_periods {
                (max_idx - start + 1).f64()
            } else { f64::NAN };
            if NotNan!({*arr.uget(start)}) {valid_window -= 1};
            start += 1;
        }
    }
}


pub fn ts_min_1d<T: Number> (arr: ArrayView1<T>, mut out: ArrayViewMut1<f64>, window: usize, min_periods: usize, _out_step: usize)
where usize: AsPrimitive<T>
{   
    assert!(window >= min_periods, "滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(out.len() == arr.len(), "输入数组的维度必须等于输出数组的维度");
    let mut min: T = T::max_();
    let mut min_idx = 0usize;
    let mut valid_window = 0usize;
    for i in 0..window-1 { // 安全性：i不会超过arr和out的长度
        let v = unsafe {*arr.uget(i)};
        if v.notnan() {valid_window += 1};
        if v < min {(min, min_idx) = (v, i);}
        unsafe {
            *out.uget_mut(i) = if valid_window >= min_periods {
                min.f64()
            } else { f64::NAN };
        }
    }
    let mut start = 0;
    for end in window-1..arr.len() {   
        // 安全性：start和end不会超过arr和out的长度
        unsafe {
            let v = *arr.uget(end);
            if v.notnan() {valid_window += 1};
            if min_idx < start { // 最小值已经失效，重新找到最小值
                let v = *arr.uget(start);
                min = if IsNan!(v) {T::max_()} else {v};
                for i in start..=end {
                    let v = *arr.uget(i);
                    if v <= min {(min, min_idx) = (v, i);}
                }
            } else if v <= min {  // 当前是最小值
                (min, min_idx) = (v, end);
            }
            *out.uget_mut(end) = if valid_window >= min_periods {
                min.f64()
            } else { f64::NAN };
            if NotNan!({*arr.uget(start)}) {valid_window -= 1};
            start += 1;
        }
    }
}

pub fn ts_argmin_1d<T: Number> (arr: ArrayView1<T>, mut out: ArrayViewMut1<f64>, window: usize, min_periods: usize, _out_step: usize)
where usize: AsPrimitive<T>
{   
    assert!(window >= min_periods, "滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(out.len() == arr.len(), "输入数组的维度必须等于输出数组的维度");
    let mut min: T = T::max_();
    let mut min_idx = 0usize;
    let mut valid_window = 0usize;
    for i in 0..window-1 { // 安全性：i不会超过arr和out的长度
        let v = unsafe {*arr.uget(i)};
        if v.notnan() {valid_window += 1};
        if v < min {(min, min_idx) = (v, i);}
        unsafe {
            *out.uget_mut(i) = if valid_window >= min_periods {
                (min_idx + 1).f64()
            } else { f64::NAN };
        }
    }
    let mut start = 0;
    for end in window-1..arr.len() {   
        // 安全性：start和end不会超过arr和out的长度
        unsafe {
            let v = *arr.uget(end);
            if v.notnan() {valid_window += 1};
            if min_idx < start { // 最小值已经失效，重新找到最小值
                let v = *arr.uget(start);
                min = if IsNan!(v) {T::max_()} else {v};
                for i in start..=end {
                    let v = *arr.uget(i);
                    if v <= min {(min, min_idx) = (v, i);}
                }
            } else if v <= min {  // 当前是最小值
                (min, min_idx) = (v, end);
            }
            *out.uget_mut(end) = if valid_window >= min_periods {
                (min_idx - start + 1).f64()
            } else { f64::NAN };
            if NotNan!({*arr.uget(start)}) {valid_window -= 1};
            start += 1;
        }
    }
}

pub fn ts_rank_1d<T: Number> (arr: ArrayView1<T>, mut out: ArrayViewMut1<f64>, window: usize, min_periods: usize, _out_step: usize)
where usize: AsPrimitive<T>
{   
    assert!(window >= min_periods, "滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(out.len() == arr.len(), "输入数组的维度必须等于输出数组的维度");
    let mut valid_window = 0usize;
    for i in 0..window-1 { // 安全性：i不会超过arr和out的长度
        unsafe {
            let v = *arr.uget(i);
            let mut n_repeat = 1; // 当前值的重复次数
            let mut rank = 1.; // 先假设为第一名，每当有元素比他更小，排名就加1
            if v.notnan() {
                valid_window += 1;
                for j in 0..i {
                    let a = *arr.uget(j);
                    if a < v {rank += 1.}
                    else if a == v {n_repeat += 1}
                }
            } else {rank = f64::NAN};
            *out.uget_mut(i) = if valid_window >= min_periods {
                rank + 0.5 * (n_repeat - 1).f64()
            } else { f64::NAN };
        }
    }
    let mut start = 0;
    for end in window-1..arr.len() {   
        // 安全性：start和end不会超过arr的长度，而out和arr长度相同
        unsafe {
            let v = *arr.uget(end);
            let mut n_repeat = 1; // 当前值的重复次数
            let mut rank = 1.; // 先假设为第一名，每当有元素比他更小，排名就加1
            if v.notnan() {
                valid_window += 1;
                for i in start..end {
                    let a = *arr.uget(i);
                    if a < v {rank += 1.}
                    else if a == v {n_repeat += 1}
                }
            } else {rank = f64::NAN};
            *out.uget_mut(end) = if valid_window >= min_periods {
                rank + 0.5 * (n_repeat - 1).f64() // 对于重复值的method: average
            } else { f64::NAN };
            let v = *arr.uget(start);
            if v.notnan() {valid_window -= 1;};
            start += 1;
        }
    }
}

pub fn ts_rank_pct_1d<T: Number> (arr: ArrayView1<T>, mut out: ArrayViewMut1<f64>, window: usize, min_periods: usize, _out_step: usize)
where usize: AsPrimitive<T>
{   
    assert!(window >= min_periods, "滚动window不能小于最小期数");
    assert!(window <= arr.len(), "滚动window不能大于数组长度");
    debug_assert!(out.len() == arr.len(), "输入数组的维度必须等于输出数组的维度");
    let mut valid_window = 0usize;
    for i in 0..window-1 { // 安全性：i不会超过arr和out的长度
        unsafe {
            let v = *arr.uget(i);
            let mut n_repeat = 1; // 当前值的重复次数
            let mut rank = 1.; // 先假设为第一名，每当有元素比他更小，排名就加1
            if v.notnan() {
                valid_window += 1;
                for j in 0..i {
                    let a = *arr.uget(j);
                    if a < v {rank += 1.}
                    else if a == v {n_repeat += 1}
                }
            } else {rank = f64::NAN};
            *out.uget_mut(i) = if valid_window >= min_periods {
                (rank + 0.5 * (n_repeat - 1).f64()) / valid_window.f64()
            } else { f64::NAN };
        }
    }
    let mut start = 0;
    for end in window-1..arr.len() {   
        // 安全性：start和end不会超过arr的长度，而out和arr长度相同
        unsafe {
            let v = *arr.uget(end);
            let mut n_repeat = 1; // 当前值的重复次数
            let mut rank = 1.; // 先假设为第一名，每当有元素比他更小，排名就加1
            if v.notnan() {
                valid_window += 1;
                for i in start..end {
                    let a = *arr.uget(i);
                    if a < v {rank += 1.}
                    else if a == v {n_repeat += 1}
                }
            } else {rank = f64::NAN};
            *out.uget_mut(end) = if valid_window >= min_periods {
                (rank + 0.5 * (n_repeat - 1).f64()) / valid_window.f64() // 对于重复值的method: average
            } else { f64::NAN };
            let v = *arr.uget(start);
            if v.notnan() {valid_window -= 1;};
            start += 1;
        }
    }
}