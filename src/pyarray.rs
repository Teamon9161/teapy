use crate::datatype::{Number, TsFunc, TsFunc2, ArrayFunc};
use numpy::ndarray::{
    ArrayView1, ArrayViewMut1,
    ArrayView2, ArrayViewMut2,
    ArrayViewD, ArrayViewMutD,
    Ix1, Ix2, IxDyn,
    Zip, Axis,
};
use numpy::{
    PyArray1, PyArrayDyn, PyReadonlyArrayDyn, PyReadwriteArrayDyn
};
use pyo3::FromPyObject;

use crate::window_func::*;
use crate::array::*;

// 只读和读写的动态python array转换为确定维度的ArrayView或可变ArrayView
pub trait DynToArrayRead {
    type Type: Number;
    fn to_array1(&self) -> ArrayView1<Self::Type>;
    fn to_array2(&self) -> ArrayView2<Self::Type>;
    fn to_arrayd(&self) -> ArrayViewD<Self::Type>;
}

macro_rules! impl_dyn_to_array_read {
    ($dtype:ty) => {
        impl DynToArrayRead for PyReadonlyArrayDyn<'_, $dtype> {
            type Type = $dtype;
            fn to_array1(&self) -> ArrayView1<$dtype> {
                self.as_array().into_dimensionality::<Ix1>().unwrap()
            }
            fn to_array2(&self) -> ArrayView2<$dtype> {
                self.as_array().into_dimensionality::<Ix2>().unwrap()
            }
            fn to_arrayd(&self) -> ArrayViewD<$dtype> {
                self.as_array().into_dimensionality::<IxDyn>().unwrap()
            }
        }
    };
}

impl_dyn_to_array_read!(f64);
impl_dyn_to_array_read!(f32);
impl_dyn_to_array_read!(i32);
impl_dyn_to_array_read!(i64);

pub trait DynToArrayWrite {
    type Type: Number;
    fn to_array1(&mut self) -> ArrayViewMut1<Self::Type>;
    fn to_array2(&mut self) -> ArrayViewMut2<Self::Type>;
    fn to_arrayd(&mut self) -> ArrayViewMutD<Self::Type>;
}

macro_rules! impl_dyn_to_array_write {
    ($dtype:ty) => {
        impl DynToArrayWrite for PyReadwriteArrayDyn<'_, $dtype> {
            type Type = $dtype;
            fn to_array1(&mut self) -> ArrayViewMut1<$dtype> {
                self.as_array_mut().into_dimensionality::<Ix1>().unwrap()
            }
            fn to_array2(&mut self) -> ArrayViewMut2<$dtype> {
                self.as_array_mut().into_dimensionality::<Ix2>().unwrap()
            }
            fn to_arrayd(&mut self) -> ArrayViewMutD<$dtype> {
                self.as_array_mut().into_dimensionality::<IxDyn>().unwrap()
            }
        }
    };
}

impl_dyn_to_array_write!(f32);
impl_dyn_to_array_write!(f64);
impl_dyn_to_array_write!(i32);
impl_dyn_to_array_write!(i64);

pub trait CallFunction<T: Number> {
    fn call_array_function<U: Number> ( // 普通array函数
        &self, 
        f: ArrayFunc<T, U>, 
        axis: Option<usize>,
        par: Option<bool>
    ) -> &PyArrayDyn<U>;

    fn call_ts_function( // 普通滚动函数
        &self, 
        f: TsFunc<T>,
        window: usize,
        axis: Option<usize>, 
        min_periods: Option<usize>, 
        par: Option<bool>
    ) -> &PyArrayDyn<f64>;

    fn call_ts_function2( // 有两个输入的滚动函数
        &self, 
        other: &PyArrayDyn<T>,
        f: TsFunc2<T>,
        window: usize,
        axis: Option<usize>, 
        min_periods: Option<usize>, 
        par: Option<bool>
    ) -> &PyArrayDyn<f64>;
}
// 根据维度调用一维的算法函数，可以并行
macro_rules! impl_call_function {
    ($dtype:ty) => {
        impl CallFunction<$dtype> for PyArrayDyn<$dtype> {
            // 普通对array一个轴应用的函数, 例如argsort，rank等, U是output的dtype
            fn call_array_function<U: Number> (&self, f: ArrayFunc<$dtype, U>, axis: Option<usize>, par: Option<bool>) -> &PyArrayDyn<U>{
                let par = par.unwrap_or(false);
                let ndim = self.ndim();
                let axis = Axis(axis.unwrap_or(0));
                let out = if ndim == 1 {
                    let x = self.readonly();
                    let x_r = x.to_array1();
                    let out = PyArray1::<U>::zeros(self.py(), x_r.dim(), false);
                    f(x_r, out.readwrite().as_array_mut());
                    out.to_dyn()  
                } else if ndim >= 2 {
                    let x = self.readonly();
                    let x_r = x.to_arrayd();
                    let f_contiguous_flag = self.is_fortran_contiguous(); // 是否是f连续
                    let out = PyArrayDyn::<U>::zeros(self.py(), x_r.dim(), f_contiguous_flag);
                    let mut out_wr = out.readwrite();
                    let mut out_wr = out_wr.as_array_mut();
                    let arr_zip = Zip::from(out_wr.lanes_mut(axis)).and(x_r.lanes(axis));
                    if !par { // 非并行
                        arr_zip.for_each(|out_wr, x_r| {
                            f(x_r, out_wr);
                        });
                    } else { // 并行
                        arr_zip.par_for_each(|out_wr, x_r| {
                            f(x_r, out_wr);
                        });
                    }
                    out
                } else {panic!("不支持0维数组")};
                out
            }
            // 接受单个array的滚动函数
            fn call_ts_function(&self, f: TsFunc<$dtype>, window: usize, axis: Option<usize>, min_periods: Option<usize>, par: Option<bool>) -> &PyArrayDyn<f64>{
                let min_periods = min_periods.unwrap_or(1); // 默认最小需要的周期为1
                let axis = Axis(axis.unwrap_or(0));
                let par = par.unwrap_or(false); // 默认不并行
                let ndim = self.ndim();
                let out = if ndim == 1 {
                    let x = self.readonly();
                    let x_r = x.to_array1();
                    let out_step = x_r.stride_of(axis) as usize; // 目前暂时无用
                    let out = PyArray1::<f64>::zeros(self.py(), x_r.dim(), false);
                    f(x_r, out.readwrite().as_array_mut(), window, min_periods, out_step);
                    out.to_dyn()  
                } else if ndim >= 2 {
                    let x = self.readonly();
                    let x_r = x.to_arrayd();
                    let out_step = 1usize;
                    let f_contiguous_flag = self.is_fortran_contiguous(); // 是否是f连续
                    let out = PyArrayDyn::<f64>::zeros(self.py(), x_r.dim(), f_contiguous_flag);
                    let mut out_wr = out.readwrite();
                    let mut out_wr = out_wr.as_array_mut();
                    let arr_zip = Zip::from(out_wr.lanes_mut(axis)).and(x_r.lanes(axis));
                    if !par { // 非并行
                        arr_zip.for_each(|out_wr, x_r| {
                            f(x_r, out_wr, window, min_periods, out_step);
                        });
                    } else { // 并行
                        arr_zip.par_for_each(|out_wr, x_r| {
                            f(x_r, out_wr, window, min_periods, out_step);
                        });
                    }
                    out
                }
                else {panic!("不支持0维数组")};
                out
            }
            // 接受两个array的滚动函数
            fn call_ts_function2(&self, other: &PyArrayDyn<$dtype>, f: TsFunc2<$dtype>, window: usize, axis: Option<usize>, min_periods: Option<usize>, par: Option<bool>) -> &PyArrayDyn<f64>{
                let min_periods = min_periods.unwrap_or(1); // 默认最小需要的周期为1
                let axis = Axis(axis.unwrap_or(0));
                let par = par.unwrap_or(false); // 默认不并行
                let ndim = self.ndim();
                let out = if ndim == 1 {
                    let (x, y) = (self.readonly(), other.readonly());
                    let (x_r, y_r) = (x.to_array1(), y.to_array1());
                    let out_step = x_r.stride_of(axis) as usize; // 目前暂时无用
                    let out = PyArray1::<f64>::zeros(self.py(), x_r.dim(), false);
                    f(x_r, y_r, out.readwrite().as_array_mut(), window, min_periods, out_step);
                    out.to_dyn()  
                } else if ndim >= 2 {
                    let (x, y) = (self.readonly(), other.readonly());
                    let (x_r, y_r) = (x.to_arrayd(), y.to_arrayd());
                    let out_step = x_r.strides()[axis.index()] as usize; // 目前暂时无用
                    let f_contiguous_flag = self.is_fortran_contiguous(); // 是否是f连续
                    let out = PyArrayDyn::<f64>::zeros(self.py(), x_r.dim(), f_contiguous_flag);
                    let mut out_wr = out.readwrite();
                    let mut out_wr = out_wr.as_array_mut();
                    let arr_zip = Zip::from(out_wr.lanes_mut(axis)).and(x_r.lanes(axis)).and(y_r.lanes(axis));
                    if !par { // 非并行
                        arr_zip.for_each(|out_wr, x_r, y_r| {
                            f(x_r, y_r, out_wr, window, min_periods, out_step);
                        });
                    } else { // 并行
                        arr_zip.par_for_each(|out_wr, x_r, y_r| {
                            f(x_r, y_r, out_wr, window, min_periods, out_step);
                        });
                    }
                    out
                }
                else {panic!("不支持0维数组")};
                out
            }
        }
    };
}

impl_call_function!(f32);
impl_call_function!(f64);
impl_call_function!(i32);
impl_call_function!(i64);

// 所有可接受的python array类型
#[derive(FromPyObject)]
pub enum PyArrayOk<'py> {
    F32(&'py PyArrayDyn<f32>),
    F64(&'py PyArrayDyn<f64>),
    I32(&'py PyArrayDyn<i32>),
    I64(&'py PyArrayDyn<i64>),
}


impl<'py> PyArrayOk<'py> {
    // 获得维度信息
    pub fn ndim(&self) -> usize {
        match self {
            PyArrayOk::F32(a) => a.ndim(),
            PyArrayOk::F64(a) => a.ndim(),
            PyArrayOk::I32(a) => a.ndim(),
            PyArrayOk::I64(a) => a.ndim(),
        }
    }
}

// 为允许的py array添加函数, 接收一个array的普通window func
macro_rules! impl_arrayfunc {
    ($name:ident, $func:ident, $otype:ty) => {
        impl<'py> PyArrayOk<'py> {
            pub fn $name (self, axis: Option<usize>, par: Option<bool>) -> &'py PyArrayDyn<$otype> {
                use PyArrayOk::*;
                match self {
                    F32(arr) => arr.call_array_function::<$otype>($func::<f32> as ArrayFunc<f32, $otype>, axis, par),
                    F64(arr) => arr.call_array_function::<$otype>($func::<f64> as ArrayFunc<f64, $otype>, axis, par),
                    I32(arr) => arr.call_array_function::<$otype>($func::<i32> as ArrayFunc<i32, $otype>, axis, par),
                    I64(arr) => arr.call_array_function::<$otype>($func::<i64> as ArrayFunc<i64, $otype>, axis, par),
                    // _ => todo!()
                }
            }
        }
    };
}

// 为允许的py array添加函数, 接收一个array的普通window func
macro_rules! impl_tsfunc {
    ($name:ident, $func:ident) => {
        impl<'py> PyArrayOk<'py> {
            pub fn $name(self, window: usize, axis: Option<usize>, min_periods: Option<usize>, par: Option<bool>) -> &'py PyArrayDyn<f64> {
                use PyArrayOk::*;
                match self {
                    F32(arr) => arr.call_ts_function($func::<f32> as TsFunc<f32>, window, axis, min_periods, par),
                    F64(arr) => arr.call_ts_function($func::<f64> as TsFunc<f64>, window, axis, min_periods, par),
                    I32(arr) => arr.call_ts_function($func::<i32> as TsFunc<i32>, window, axis, min_periods, par),
                    I64(arr) => arr.call_ts_function($func::<i64> as TsFunc<i64>, window, axis, min_periods, par),
                    // _ => todo!()
                }
            }
        }
    };
}

// 为允许的py array添加函数, 接收两个array的window func
macro_rules! impl_tsfunc2 {
    ($name:ident, $func:ident) => {
        impl<'py> PyArrayOk<'py> {
            pub fn $name(self, other: &PyArrayOk, window: usize, axis: Option<usize>, min_periods: Option<usize>, par: Option<bool>) -> &'py PyArrayDyn<f64> {
                use PyArrayOk::*;
                match (self, other) {
                    (F32(arr), F32(other)) => arr.call_ts_function2(other, $func::<f32> as TsFunc2<f32>, window, axis, min_periods, par),
                    (F64(arr), F64(other)) => arr.call_ts_function2(other, $func::<f64> as TsFunc2<f64>, window, axis, min_periods, par),
                    (I32(arr), I32(other)) => arr.call_ts_function2(other, $func::<i32> as TsFunc2<i32>, window, axis, min_periods, par),
                    (I64(arr), I64(other)) => arr.call_ts_function2(other, $func::<i64> as TsFunc2<i64>, window, axis, min_periods, par),
                    _ => panic!("左右两边array的类型不匹配")
                }
            }
        }
    };
}

// array function
impl_arrayfunc!(argsort, argsort_1d, usize);
impl_arrayfunc!(rank, rank_1d, f64);
impl_arrayfunc!(rank_pct, rank_pct_1d, f64);

// feature
impl_tsfunc!(ts_sma, ts_sma_1d);
impl_tsfunc!(ts_ewm, ts_ewm_1d);
impl_tsfunc!(ts_wma, ts_wma_1d);
impl_tsfunc!(ts_sum, ts_sum_1d);
impl_tsfunc!(ts_prod, ts_prod_1d);
impl_tsfunc!(ts_prod_mean, ts_prod_mean_1d);
impl_tsfunc!(ts_std, ts_std_1d);
impl_tsfunc!(ts_skew, ts_skew_1d);
impl_tsfunc!(ts_kurt, ts_kurt_1d);

// compare
impl_tsfunc!(ts_max, ts_max_1d);
impl_tsfunc!(ts_min, ts_min_1d);
impl_tsfunc!(ts_argmax, ts_argmax_1d);
impl_tsfunc!(ts_argmin, ts_argmin_1d);
impl_tsfunc!(ts_rank, ts_rank_1d);
impl_tsfunc!(ts_rank_pct, ts_rank_pct_1d);

// norm
impl_tsfunc!(ts_stable, ts_stable_1d);
impl_tsfunc!(ts_minmaxnorm, ts_minmaxnorm_1d);
impl_tsfunc!(ts_meanstdnorm, ts_meanstdnorm_1d);

// cov corr
impl_tsfunc2!(ts_cov, ts_cov_1d);
impl_tsfunc2!(ts_corr, ts_corr_1d);

// reg
impl_tsfunc!(ts_reg, ts_reg_1d);
impl_tsfunc!(ts_tsf, ts_tsf_1d);
impl_tsfunc!(ts_reg_slope, ts_reg_slope_1d);
impl_tsfunc!(ts_reg_intercept, ts_reg_intercept_1d);