use std::ops::{Add, Sub};
/// Fast2sum is a floating-point algorithm for computing the exact
/// round-off error in a floating-point addition operation.
///
/// see https://en.wikipedia.org/wiki/2Sum
#[inline(always)]
pub fn fast2sum(sum: f64, y: f64) -> (f64, f64) {
    let t = sum + y;
    let c = (t - sum) - y;
    (t, c)
}

/// Kahan summation, see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
#[inline(always)]
pub fn kh_sum<T>(sum: T, v: T, c: &mut T) -> T
where
    T: Add<Output = T> + Sub<Output = T> + Copy,
{
    let y = v - *c;
    let t = sum + y;
    *c = (t - sum) - y;
    t
}
