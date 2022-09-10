#[macro_use]
pub(crate) mod macros;
pub(crate) use macros::*;

use crate::arrview::{ArrView1, ArrViewMut1};
use crate::datatype::Number;
use crate::pyarray::DynToArrayRead;
use numpy::ndarray::{Axis, Zip};
use numpy::PyArrayDyn;
use std::ops::IndexMut;

add_functype!(
    impl_py_array_func,
    crate::array_func,
    ArrayFunc,
    CallArrayFunc,
    call_array_func,
    -func_p (bool),
    -p (
        axis: Option<usize>,
        stable: Option<bool>,
        par: Option<bool>
    ),
    -sig "(x, axis, stable, par, /)",
    (self, f),
    {
        let ndim = self.ndim();
        let par = par.unwrap_or(false);
        let stable = stable.unwrap_or(false);
        let axis = Axis(axis.unwrap_or(0));
        let out = if ndim >= 1 {
            let x = self.readonly();
            let x_r = x.to_arrayd();
            let f_contiguous_flag = self.is_fortran_contiguous(); // 是否是f连续
            let out = PyArrayDyn::<U>::zeros(self.py(), x_r.dim(), f_contiguous_flag);
            let mut out_wr = out.readwrite();
            let mut out_wr = out_wr.as_array_mut();
            let arr_zip = Zip::from(out_wr.lanes_mut(axis)).and(x_r.lanes(axis));
            if !par || (ndim == 1) {
                // 非并行
                arr_zip.for_each(|out_wr, x_r| {
                    f(ArrView1(x_r), ArrViewMut1(out_wr), stable);
                });
            } else {
                // 并行
                arr_zip.par_for_each(|out_wr, x_r| {
                    f(ArrView1(x_r), ArrViewMut1(out_wr), stable);
                });
            }
            out
        } else {
            panic!("不支持0维数组")
        };
        out
    }
);

add_functype!(
    impl_py_rank_func,
    crate::array_func,
    RankFunc,
    CallRankFunc,
    call_rank_func,
    -func_p (bool),
    -p (
        axis: Option<usize>,
        pct: Option<bool>,
        par: Option<bool>
    ),
    -sig "(x, axis, pct, par, /)",
    (self, f),
    {
        let ndim = self.ndim();
        let par = par.unwrap_or(false);
        let pct = pct.unwrap_or(false);
        let axis = Axis(axis.unwrap_or(0));
        let out = if ndim >= 1 {
            let x = self.readonly();
            let x_r = x.to_arrayd();
            let f_contiguous_flag = self.is_fortran_contiguous(); // 是否是f连续
            let out = PyArrayDyn::<U>::zeros(self.py(), x_r.dim(), f_contiguous_flag);
            let mut out_wr = out.readwrite();
            let mut out_wr = out_wr.as_array_mut();
            let arr_zip = Zip::from(out_wr.lanes_mut(axis)).and(x_r.lanes(axis));
            if !par || (ndim == 1) {
                // 非并行
                arr_zip.for_each(|out_wr, x_r| {
                    f(ArrView1(x_r), ArrViewMut1(out_wr), pct);
                });
            } else {
                // 并行
                arr_zip.par_for_each(|out_wr, x_r| {
                    f(ArrView1(x_r), ArrViewMut1(out_wr), pct);
                });
            }
            out
        } else {
            panic!("不支持0维数组")
        };
        out
    }
);

add_functype!(
    impl_py_array_agg_func,
    crate::array_func,
    ArrayAggFunc,
    CallArrayAggFunc,
    call_array_agg_func,
    -func_p (bool),
    -p (
        axis: Option<usize>,
        stable: Option<bool>,
        par: Option<bool>,
        keepdims: Option<bool>
    ),
    -sig "(x, axis, stable, par, keepdims, /)",
    (self, f),
    {
        let par = par.unwrap_or(false);
        let axis = Axis(axis.unwrap_or(0));
        let keepdims = keepdims.unwrap_or(false);
        let stable = stable.unwrap_or(false);
        let ndim = self.ndim();
        let out = if ndim >= 1 {
            let x = self.readonly();
            let x_r = x.to_arrayd();
            let f_contiguous_flag = self.is_fortran_contiguous(); // 是否是f连续
            let mut out_dim = x_r.raw_dim();
            if !keepdims {
                *out_dim.index_mut(axis.index()) = 1;
            }
            let out = PyArrayDyn::<U>::zeros(self.py(), out_dim, f_contiguous_flag);
            let mut out_wr = out.readwrite();
            let mut out_wr = out_wr.as_array_mut();
            let arr_zip = Zip::from(out_wr.lanes_mut(axis)).and(x_r.lanes(axis));
            if !par || (ndim == 1) {
                // 非并行
                arr_zip.for_each(|out_wr, x_r| {
                    f(ArrView1(x_r), ArrViewMut1(out_wr), stable);
                });
            } else {
                // 并行
                arr_zip.par_for_each(|out_wr, x_r| {
                    f(ArrView1(x_r), ArrViewMut1(out_wr), stable);
                });
            }
            out
        } else {
            panic!("不支持0维数组")
        };
        out
    }
);

add_functype!(
    impl_py_ts_func,
    crate::window_func,
    TsFunc,
    CallTsFunc,
    call_ts_func,
    -func_p (usize, usize, bool),
    -p (
        window: usize,
        axis: Option<usize>,
        min_periods: Option<usize>,
        stable: Option<bool>,
        par: Option<bool>
    ),
    -sig "(x, window, axis, min_periods, par, /)",
    (self, f),
    {
        let ndim = self.ndim();
        let min_periods = min_periods.unwrap_or(1); // 默认最小需要的周期为1
        let axis = Axis(axis.unwrap_or(0));
        let stable = stable.unwrap_or(false);
        let par = par.unwrap_or(false); // 默认不并行

        let out = if ndim >= 1 {
            let x = self.readonly();
            let x_r = x.to_arrayd();
            let f_contiguous_flag = self.is_fortran_contiguous(); // 是否是f连续
            let out = PyArrayDyn::<U>::zeros(self.py(), x_r.dim(), f_contiguous_flag);
            let mut out_wr = out.readwrite();
            let mut out_wr = out_wr.as_array_mut();
            let arr_zip = Zip::from(out_wr.lanes_mut(axis)).and(x_r.lanes(axis));
            if !par || (ndim == 1) {
                // 非并行
                arr_zip.for_each(|out_wr, x_r| {
                    f(ArrView1(x_r), ArrViewMut1(out_wr), window, min_periods, stable);
                });
            } else {
                // 并行
                arr_zip.par_for_each(|out_wr, x_r| {
                    f(ArrView1(x_r), ArrViewMut1(out_wr), window, min_periods, stable);
                });
            }
            out
        } else {
            panic!("不支持0维数组")
        };
        out
    }
);

add_functype2!(
    impl_py_ts_func2,
    crate::window_func,
    TsFunc2,
    CallTsFunc2,
    call_ts_func2,
    -func_p (usize, usize, bool),
    -p (
        window: usize,
        axis: Option<usize>,
        min_periods: Option<usize>,
        stable: Option<bool>,
        par: Option<bool>
    ),
    -sig "(x, y, window, axis, min_periods, par, /)",
    (self, other, f),
    {
        let min_periods = min_periods.unwrap_or(1); // 默认最小需要的周期为1
        let axis = Axis(axis.unwrap_or(0));
        let stable = stable.unwrap_or(false);
        let par = par.unwrap_or(false); // 默认不并行
        let ndim = self.ndim();
        let out = if ndim >= 1 {
            let (x, y) = (self.readonly(), other.readonly());
            let (x_r, y_r) = (x.to_arrayd(), y.to_arrayd());
            let f_contiguous_flag = self.is_fortran_contiguous(); // 是否是f连续
            let out = PyArrayDyn::<U>::zeros(self.py(), x_r.dim(), f_contiguous_flag);
            let mut out_wr = out.readwrite();
            let mut out_wr = out_wr.as_array_mut();
            let arr_zip = Zip::from(out_wr.lanes_mut(axis))
                .and(x_r.lanes(axis))
                .and(y_r.lanes(axis));
            if !par || (ndim == 1) {
                // 非并行
                arr_zip.for_each(|out_wr, x_r, y_r| {
                    f(ArrView1(x_r), ArrView1(y_r), ArrViewMut1(out_wr), window, min_periods, stable);
                });
            } else {
                // 并行
                arr_zip.par_for_each(|out_wr, x_r, y_r| {
                    f(ArrView1(x_r), ArrView1(y_r), ArrViewMut1(out_wr), window, min_periods, stable);
                });
            }
            out
        } else {
            panic!("不支持0维数组")
        };
        out
    }
);

add_functype2!(
    impl_py_array_agg_func2,
    crate::array_func,
    ArrayAggFunc2,
    CallArrayAggFunc2,
    call_array_agg_func2,
    -func_p (bool),
    -p (
        axis: Option<usize>,
        stable: Option<bool>,
        par: Option<bool>,
        keepdims: Option<bool>
    ),
    -sig "(x, y, axis, stable, par, keepdims, /)",
    (self, other, f),
    {
        let axis = Axis(axis.unwrap_or(0));
        let stable = stable.unwrap_or(false);
        let par = par.unwrap_or(false); // 默认不并行
        let keepdims = keepdims.unwrap_or(false);
        let ndim = self.ndim();
        let out = if ndim >= 1 {
            let (x, y) = (self.readonly(), other.readonly());
            let (x_r, y_r) = (x.to_arrayd(), y.to_arrayd());
            let f_contiguous_flag = self.is_fortran_contiguous(); // 是否是f连续
            let mut out_dim = x_r.raw_dim();
            if !keepdims {
                *out_dim.index_mut(axis.index()) = 1;
            }
            let out = PyArrayDyn::<U>::zeros(self.py(), out_dim, f_contiguous_flag);
            let mut out_wr = out.readwrite();
            let mut out_wr = out_wr.as_array_mut();
            let arr_zip = Zip::from(out_wr.lanes_mut(axis))
                .and(x_r.lanes(axis))
                .and(y_r.lanes(axis));
            if !par || (ndim == 1) {
                // 非并行
                arr_zip.for_each(|out_wr, x_r, y_r| {
                    f(ArrView1(x_r), ArrView1(y_r), ArrViewMut1(out_wr), stable);
                });
            } else {
                // 并行
                arr_zip.par_for_each(|out_wr, x_r, y_r| {
                    f(ArrView1(x_r), ArrView1(y_r), ArrViewMut1(out_wr), stable);
                });
            }
            out
        } else {
            panic!("不支持0维数组")
        };
        out
    }
);
