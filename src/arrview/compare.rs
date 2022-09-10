use super::prelude::*;
use ndarray::Array1;
use std::cmp::Ordering;

impl_arrview!([ArrView1, ArrViewMut1], Number, {
    pub fn argsort(&self, out: &mut ArrViewMut1<i32>) {
        assert!(out.len() >= self.len());
        let mut i = 0;
        out.apply_mut(|v| {
            *v = i;
            i += 1;
        }); // set elements of out array
        out.sort_unstable_by(|a, b| {
            let (va, vb) = unsafe { (*self.uget((*a) as usize), *self.uget((*b) as usize)) }; // safety: out不超过self的长度
            if vb.isnan() | (va < vb) {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        });
    }

    #[allow(unused_assignments)]
    pub fn rank(&self, out: &mut ArrViewMut1<f64>, pct: bool) {
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
        let mut idx_sorted = Array1::from_iter(0..len);
        ArrViewMut1(idx_sorted.view_mut()).sort_unstable_by(|a, b| {
            let (va, vb) = unsafe { (*self.uget((*a) as usize), *self.uget((*b) as usize)) }; // safety: out不超过self的长度
            if vb.isnan() | (va < vb) {
                Ordering::Less
            } else {
                Ordering::Greater
            }
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
            let notnan_count = self.count_notnan();
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
});
