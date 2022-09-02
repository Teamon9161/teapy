use crate::datatype::Number;
use numpy::ndarray::{Array, ArrayView1, ArrayViewMut1};
use std::cmp::Ordering;
use std::iter::zip;

pub fn argsort_1d<T: Number>(arr: ArrayView1<T>, mut out: ArrayViewMut1<usize>) {
    let mut i = 0;
    let c_flag = out.is_standard_layout();
    if c_flag {
        for v in &mut out {
            *v = i;
            i += 1;
        } // 更改out的数值
        let out_slice = out.as_slice_mut().unwrap();
        out_slice.sort_unstable_by(|a, b| {
            let (va, vb) = unsafe { (*arr.uget(*a), *arr.uget(*b)) }; // safety: out不超过self的长度
            if vb.isnan() | (va < vb) {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        });
    } else {
        let mut out_c = Array::from_iter(0..arr.len());
        let out_slice = out_c.as_slice_mut().unwrap();
        out_slice.sort_unstable_by(|a, b| {
            let (va, vb) = unsafe { (*arr.uget(*a), *arr.uget(*b)) }; // safety: out不超过self的长度
            if vb.isnan() | (va < vb) {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        });
        for (v, v_out) in zip(out_slice, out) {
            *v_out = *v
        }
    }
}

#[allow(unused_assignments)]
pub fn rank_1d<T: Number>(arr: ArrayView1<T>, mut out: ArrayViewMut1<f64>) {
    let len = arr.len();
    if len == 0 {
        return;
    } else if len == 1 {
        unsafe { *out.uget_mut(0) = 1. };
        return;
    }
    let mut idx_sorted = Array::from_iter(0..arr.len());
    idx_sorted.as_slice_mut().unwrap().sort_unstable_by(|a, b| {
        let (va, vb) = unsafe { (*arr.uget(*a), *arr.uget(*b)) }; // safety: out不超过self的长度
        if vb.isnan() | (va < vb) {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    });
    // 如果最小值是nan说明全是nan
    if unsafe { *arr.uget(*idx_sorted.uget(0)) }.isnan() {
        out.mapv_inplace(|_| f64::NAN);
        return;
    }
    let mut repeat_num = 1usize;
    let mut nan_flag = false;
    let (mut cur_rank, mut sum_rank) = (1usize, 0usize);
    let (mut idx, mut idx1) = (0, 0);
    unsafe {
        for i in 0..len - 1 {
            // safe because i_max = arr.len()-2 and arr.len() >= 2
            (idx, idx1) = (*idx_sorted.uget(i), *idx_sorted.uget(i + 1));
            let (v, v1) = (*arr.uget(idx), *arr.uget(idx1)); // 下一个值，safe because idx1 < arr.len()
            if v1.isnan() {
                // 下一个值是nan，说明后面的值全是nan
                sum_rank += cur_rank;
                cur_rank += 1;
                for j in 0..repeat_num {
                    // safe because i >= repeat_num
                    *out.uget_mut(*idx_sorted.uget(i - j)) = sum_rank.f64() / repeat_num.f64()
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
                    *out.uget_mut(*idx_sorted.uget(i - j)) = sum_rank.f64() / repeat_num.f64()
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
}

pub fn count_nan<T: Number>(arr: ArrayView1<T>) -> usize {
    let mut nan_count = 0;
    for v in &arr {
        if v.isnan() {
            nan_count += 1;
        }
    }
    nan_count
}

pub fn count_notnan<T: Number>(arr: ArrayView1<T>) -> usize {
    let nan_count = count_nan(arr);
    arr.len() - nan_count
}

#[allow(unused_assignments)]
pub fn rank_pct_1d<T: Number>(arr: ArrayView1<T>, mut out: ArrayViewMut1<f64>) {
    let len = arr.len();
    if len == 0 {
        return;
    } else if len == 1 {
        unsafe { *out.uget_mut(0) = 1. };
        return;
    }
    let mut idx_sorted = Array::from_iter(0..arr.len());
    idx_sorted.as_slice_mut().unwrap().sort_unstable_by(|a, b| {
        let (va, vb) = unsafe { (*arr.uget(*a), *arr.uget(*b)) }; // safety: out不超过self的长度
        if vb.isnan() | (va < vb) {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    });
    // 如果最小值是nan说明全是nan
    if unsafe { *arr.uget(*idx_sorted.uget(0)) }.isnan() {
        out.mapv_inplace(|_| f64::NAN);
        return;
    }

    let notnan_count = count_notnan(arr);
    let mut repeat_num = 1usize;
    let mut nan_flag = false;
    let (mut cur_rank, mut sum_rank) = (1usize, 0usize);
    let (mut idx, mut idx1) = (0, 0);
    unsafe {
        for i in 0..len - 1 {
            // safe because i_max = arr.len()-2 and arr.len() >= 2
            (idx, idx1) = (*idx_sorted.uget(i), *idx_sorted.uget(i + 1));
            let (v, v1) = (*arr.uget(idx), *arr.uget(idx1)); // 下一个值，safe because idx1 < arr.len()
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

pub fn stable_kurt_1d<T: Number>(arr: ArrayView1<T>, out: &mut f64)
where
    usize: Number,
{
    let mut n = 0usize;
    let (mut mean, mut m2, mut m3, mut m4) = (0f64, 0f64, 0f64, 0f64);

    for v in arr {
        let n1 = n;
        n += 1;
        let delta = v.f64() - mean;
        let delta_n = delta / n.f64();
        let delta_n2 = delta_n.powi(2);
        let term1 = delta * delta_n * n1.f64();
        mean += delta_n;
        m4 += term1 * delta_n2 * (n * n - 3 * n + 3).f64() + 6. * delta_n2 * m2 - 4. * delta_n * m3;
        m3 += term1 * delta_n * (n.f64() - 2.f64()) - 3. * delta_n * m2;
        m2 += term1
    }
    let n = n.f64();
    *out = if m2 != 0. {
        // (n.f64() * m4) / m2.powi(2) - 3.
        1. / ((n - 2.) * (n - 3.))
            * ((n.powi(2) - 1.) * (n * m4) / m2.powi(2) - 3. * (n - 1.).powi(2))
    } else {
        f64::NAN
    }
}
