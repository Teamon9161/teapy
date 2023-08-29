//! impl methods that return an array.

// use crate::lazy::context::Context;
use crate::lazy::{Expr, ExprElement, RefType};
use crate::{
    ArbArray, Arr, Arr1, Cast, CollectTrustedToVec, DateTime, Exprs, FillMethod, GetNone, Number,
    OptUsize, PyValue, TpResult, WrapNdarray,
};
use ndarray::{Array1, ArrayViewD, Axis, Zip};
use num::traits::real::Real;
use num::{Float, One, Signed, Zero};
use pyo3::Python;
use rayon::prelude::*;
#[cfg(feature = "stat")]
use statrs::distribution::ContinuousCDF;
use std::cmp::Ordering;
use std::mem;

pub enum DropNaMethod {
    Any,
    All,
}

impl<'a, T> Expr<'a, T>
where
    T: ExprElement + 'a,
{
    pub fn hint_arr_type(self) -> Expr<'a, &'a str> {
        self.chain_arr_f(
            move |arb_arr| {
                use ArbArray::*;
                let out: Vec<&'a str> = match arb_arr {
                    View(_) => vec!["View"],
                    ViewMut(_) => vec!["ViewMut"],
                    Owned(_) => vec!["Owned"],
                };
                Ok(ArbArray::Owned(Arr1::from_vec(out).to_dimd()))
            },
            RefType::False,
        )
    }

    pub fn full(shape: Expr<'a, usize>, value: Expr<'a, T>) -> Expr<'a, T>
    where
        T: Clone,
    {
        shape.chain_view_f_ct(
            |(sh, ct)| {
                let ndim = sh.ndim();
                let (value, ct) = value.eval(ct)?;
                let v = value
                    .view_arr()
                    .to_dim0()?
                    // .expect("value should be dim 0")
                    .into_scalar();
                if ndim == 0 {
                    let shape = sh.to_dim0()?.into_scalar();
                    Ok((Arr::from_elem(*shape, v.clone()).to_dimd().into(), ct))
                } else if ndim == 1 {
                    let shape = sh.to_dim1()?;
                    Ok((
                        Arr::from_elem(shape.to_slice().unwrap(), v.clone())
                            .to_dimd()
                            .into(),
                        ct,
                    ))
                } else {
                    Err("the dim of shape should not be greater than 1".into())
                }
            },
            RefType::False,
        )
    }

    pub fn arange(start: Option<Expr<'a, T>>, end: Expr<'a, T>, step: Option<T>) -> Expr<'a, T>
    where
        T: Float + Zero + One,
    {
        end.chain_owned_f_ct(move |(arr, ct)| {
            let start = start.map(|s| {
                *s.eval(ct.clone())
                    .unwrap()
                    .0
                    .view_arr()
                    .to_dim0()
                    .unwrap()
                    .into_scalar()
            });
            let end = arr.to_dim0()?.0.into_scalar();
            Ok((
                Array1::range(
                    start.unwrap_or_else(T::zero),
                    end,
                    step.unwrap_or_else(T::one),
                )
                .wrap()
                .to_dimd()
                .into(),
                ct,
            ))
        })
    }

    pub fn count_v(self, value: T, axis: i32, par: bool) -> Expr<'a, i32>
    where
        T: Eq + Clone,
    {
        self.chain_view_f(
            move |arr| Ok(arr.count_v(value, axis, par).into()),
            RefType::False,
        )
    }

    // deep clone expression
    pub fn deep_copy(self) -> Self
    where
        T: Clone,
    {
        self.chain_view_f(move |arr| Ok(arr.to_owned().into()), RefType::False)
    }

    pub fn fillna<T2>(self, method: FillMethod, value: Option<T2>, axis: i32, par: bool) -> Self
    where
        T: Number,
        T2: Cast<T> + Clone + Send + Sync + 'a,
        f64: Cast<T>,
    {
        self.chain_arr_f(
            move |arb_arr| {
                use ArbArray::*;
                match arb_arr {
                    View(arr) => Ok(arr.fillna(method, value, axis, par).into()),
                    ViewMut(mut arr) => {
                        arr.fillna_inplace(method, value, axis, par);
                        Ok(ViewMut(arr))
                    }
                    Owned(mut arr) => {
                        arr.view_mut().fillna_inplace(method, value, axis, par);
                        Ok(Owned(arr))
                    }
                }
            },
            RefType::Keep,
        )
    }

    pub fn clip<T2, T3>(self, min: T2, max: T3, axis: i32, par: bool) -> Self
    where
        T: Number,
        T2: Number + Cast<T>,
        T3: Number + Cast<T>,
    {
        // self.chain_view_f(move |arr| arr.clip(min, max, axis, par).into())
        self.chain_arr_f(
            move |arb_arr| {
                use ArbArray::*;
                match arb_arr {
                    View(arr) => Ok(arr.clip(min, max, axis, par).into()),
                    ViewMut(mut arr) => {
                        arr.clip_inplace(min, max, axis, par);
                        Ok(ViewMut(arr))
                    }
                    Owned(mut arr) => {
                        arr.view_mut().clip_inplace(min, max, axis, par);
                        Ok(Owned(arr))
                    }
                }
            },
            RefType::Keep,
        )
    }

    pub fn shift(self, n: i32, fill: Option<Expr<'a, T>>, axis: i32, par: bool) -> Self
    where
        T: Clone + GetNone,
    {
        self.chain_view_f_ct(
            move |(arr, ct)| {
                Ok((
                    arr.shift(
                        n,
                        fill.map(|fill| {
                            fill.eval(ct.clone())
                                .unwrap()
                                .0
                                .view_arr()
                                .to_dim0()
                                .unwrap()
                                .into_scalar()
                                .clone()
                        })
                        .unwrap_or_else(|| T::none()),
                        axis,
                        par,
                    )
                    .into(),
                    ct,
                ))
            },
            RefType::False,
        )
    }

    /// Returns the square root of a number.
    ///
    /// Returns NaN if self is a negative number other than -0.0.
    pub fn sqrt(self) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().sqrt()).into()),
            RefType::False,
        )
    }

    /// Returns the cube root of each element.
    pub fn cbrt(self) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().cbrt()).into()),
            RefType::False,
        )
    }

    /// Returns the natural logarithm of each element.
    pub fn ln(self) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().ln()).into()),
            RefType::False,
        )
    }

    /// Returns ln(1+n) (natural logarithm) more accurately than if the operations were performed separately.
    pub fn ln_1p(self) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().ln_1p()).into()),
            RefType::False,
        )
    }

    /// Returns the base 2 logarithm of each element.
    pub fn log2(self) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().log2()).into()),
            RefType::False,
        )
    }

    /// Returns the base 10 logarithm of each element.
    pub fn log10(self) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().log10()).into()),
            RefType::False,
        )
    }

    /// Returns e^(self) of each element, (the exponential function).
    pub fn exp(self) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().exp()).into()),
            RefType::False,
        )
    }

    /// Returns 2^(self) of each element.
    pub fn exp2(self) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().exp2()).into()),
            RefType::False,
        )
    }

    /// Returns e^(self) - 1 in a way that is accurate even if the number is close to zero.
    pub fn exp_m1(self) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().exp_m1()).into()),
            RefType::False,
        )
    }

    /// Computes the arccosine of each element. Return value is in radians in the range 0,
    /// pi or NaN if the number is outside the range -1, 1.
    pub fn acos(self) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().acos()).into()),
            RefType::False,
        )
    }

    /// Computes the arcsine of each element. Return value is in radians in the range -pi/2,
    /// pi/2 or NaN if the number is outside the range -1, 1.
    pub fn asin(self) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().asin()).into()),
            RefType::False,
        )
    }

    /// Computes the arctangent of each element. Return value is in radians in the range -pi/2, pi/2;
    pub fn atan(self) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().atan()).into()),
            RefType::False,
        )
    }

    /// Computes the sine of each element (in radians).
    pub fn sin(self) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().sin()).into()),
            RefType::False,
        )
    }

    /// Computes the cosine of each element (in radians).
    pub fn cos(self) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().cos()).into()),
            RefType::False,
        )
    }

    /// Computes the tangent of each element (in radians).
    pub fn tan(self) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().tan()).into()),
            RefType::False,
        )
    }

    /// Returns the smallest integer greater than or equal to `self`.
    pub fn ceil(self) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().ceil()).into()),
            RefType::False,
        )
    }

    /// Returns the largest integer less than or equal to `self`.
    pub fn floor(self) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().floor()).into()),
            RefType::False,
        )
    }

    /// Returns the fractional part of each element.
    pub fn fract(self) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().fract()).into()),
            RefType::False,
        )
    }

    /// Returns the integer part of each element. This means that non-integer numbers are always truncated towards zero.
    pub fn trunc(self) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().trunc()).into()),
            RefType::False,
        )
    }

    /// Returns true if this number is neither infinite nor NaN
    pub fn is_finite(self) -> Expr<'a, bool>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().is_finite()).into()),
            RefType::False,
        )
    }

    /// Returns true if this value is positive infinity or negative infinity, and false otherwise.
    pub fn is_inf(self) -> Expr<'a, bool>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().is_infinite()).into()),
            RefType::False,
        )
    }

    /// Returns the logarithm of the number with respect to an arbitrary base.
    ///
    /// The result might not be correctly rounded owing to implementation details;
    /// `self.log2()` can produce more accurate results for base 2,
    /// and `self.log10()` can produce more accurate results for base 10.    
    pub fn log(self, base: f64) -> Expr<'a, f64>
    where
        T: Number,
    {
        self.chain_view_f(
            move |arr| Ok(arr.mapv(|v| v.f64().log(base)).into()),
            RefType::False,
        )
    }

    pub fn valid_last(self, axis: i32, par: bool) -> Self
    where
        T: Number,
        f64: Cast<T>,
    {
        self.chain_view_f(
            move |arr| Ok(arr.valid_last(axis, par).into()),
            RefType::False,
        )
    }

    pub fn valid_first(self, axis: i32, par: bool) -> Self
    where
        T: Number,
        f64: Cast<T>,
    {
        self.chain_view_f(
            move |arr| Ok(arr.valid_first(axis, par).into()),
            RefType::False,
        )
    }

    pub fn abs(self, par: bool) -> Self
    where
        T: Signed + Clone,
    {
        self.chain_view_f(move |arr| Ok(arr.abs(par).into()), RefType::False)
    }

    pub fn sign(self, par: bool) -> Self
    where
        T: Signed + Clone,
    {
        self.chain_view_f(move |arr| Ok(arr.sign(par).into()), RefType::False)
    }

    pub fn powi(self, other: Expr<'a, i32>, par: bool) -> Self
    where
        T: Real,
    {
        self.chain_view_f_ct(
            move |(arr, ct)| {
                Ok((
                    arr.powi(&other.eval(ct.clone())?.0.view_arr(), par).into(),
                    ct,
                ))
            },
            RefType::False,
        )
    }

    pub fn powf(self, other: Expr<'a, T>, par: bool) -> Self
    where
        T: Real,
    {
        self.chain_view_f_ct(
            move |(arr, ct)| {
                Ok((
                    arr.powf(&other.eval(ct.clone())?.0.view_arr(), par).into(),
                    ct,
                ))
            },
            RefType::False,
        )
    }

    #[cfg(feature = "stat")]
    pub fn t_cdf(self, df: Expr<'a, f64>, loc: Option<f64>, scale: Option<f64>) -> Expr<'a, f64>
    where
        T: Number,
    {
        use statrs::distribution::StudentsT;
        self.chain_view_f_ct(
            move |(arr, ct)| {
                let (df, ct) = df.eval(ct)?;
                let df = *df.view_arr().to_dim0()?.into_scalar();
                let n = StudentsT::new(loc.unwrap_or(0.), scale.unwrap_or(1.), df).unwrap();
                Ok((arr.map(|v| n.cdf(v.f64())).into(), ct))
            },
            RefType::False,
        )
    }

    #[cfg(feature = "stat")]
    pub fn norm_cdf(self, mean: Option<f64>, std: Option<f64>) -> Expr<'a, f64>
    where
        T: Number,
    {
        use statrs::distribution::Normal;
        self.chain_view_f(
            move |arr| {
                let n = Normal::new(mean.unwrap_or(0.), std.unwrap_or(1.)).unwrap();
                Ok(arr.map(|v| n.cdf(v.f64())).into())
            },
            RefType::False,
        )
    }

    #[cfg(feature = "stat")]
    pub fn f_cdf(self, df1: Expr<'a, f64>, df2: Expr<'a, f64>) -> Expr<'a, f64>
    where
        T: Number,
    {
        use statrs::distribution::FisherSnedecor;
        self.chain_view_f_ct(
            move |(arr, ct)| {
                let df1 = *df1.eval(ct.clone())?.0.view_arr().to_dim0()?.into_scalar();
                let df2 = *df2.eval(ct.clone())?.0.view_arr().to_dim0()?.into_scalar();
                let n = FisherSnedecor::new(df1, df2).unwrap();
                Ok((arr.map(|v| n.cdf(v.f64())).into(), ct))
            },
            RefType::False,
        )
    }

    pub fn filter(self, mask: Expr<'a, bool>, axis: Expr<'a, i32>, par: bool) -> Self
    where
        T: Clone,
    {
        self.chain_view_f_ct(
            move |(arr, ct)| {
                Ok((
                    arr.filter(
                        &mask.eval(ct.clone())?.0.view_arr().to_dim1()?,
                        // .expect("mask should be dim 1"),
                        *axis.eval(ct.clone())?.0.view_arr().to_dim0()?.into_scalar(),
                        par,
                    )
                    .into(),
                    ct,
                ))
            },
            RefType::False,
        )
    }

    pub fn dropna<'b: 'a>(self, axis: Expr<'a, i32>, how: DropNaMethod, par: bool) -> Self
    where
        T: Clone + Number,
    {
        self.chain_view_f_ct(
            move |(arr, ct)| {
                let ndim = arr.ndim();
                let axis = *axis.eval(ct.clone())?.0.view_arr().to_dim0()?.into_scalar();
                match ndim {
                    1 => Ok((
                        arr.to_dim1()?
                            // .unwrap()
                            .remove_nan_1d()
                            .to_dimd()
                            .into(),
                        ct,
                    )),
                    2 => {
                        let arr = arr.to_dim2()?;
                        let axis_n = arr.norm_axis(axis);
                        let mask = match (axis_n, how) {
                            (Axis(0), DropNaMethod::Any) => arr.not_nan().all(1, par),
                            (Axis(0), DropNaMethod::All) => arr.not_nan().any(1, par),
                            (Axis(1), DropNaMethod::Any) => arr.not_nan().all(0, par),
                            (Axis(1), DropNaMethod::All) => arr.not_nan().any(0, par),
                            _ => panic!("axis should be 0 or 1 and how should be any or all"),
                        };
                        Ok((arr.filter(&mask, axis, par).to_dimd().into(), ct))
                    }
                    dim => Err(format!(
                        "dropna only support 1d and 2d array currently, but the array is dim {dim}"
                    )
                    .into()),
                }
            },
            RefType::False,
        )
    }

    /// select on a given axis by a slice expression
    pub fn select_by_expr(self, slc: Expr<'a, usize>, axis: Expr<'a, i32>) -> Self
    where
        T: Clone,
    {
        self.chain_view_f_ct(
            move |(arr, ct)| {
                let (slc, ct) = slc.no_dim0().eval(ct)?;
                let slc_eval = slc.view_arr();
                let axis = *axis.eval(ct.clone())?.0.view_arr().to_dim0()?.into_scalar();
                if slc_eval.ndim() > 1 {
                    return Err("The slice must be dim 0 or dim 1 when select on axis".into());
                }
                let axis = arr.norm_axis(axis);

                if slc_eval.len() == 1 {
                    Ok((
                        arr.index_axis(axis, slc_eval.to_dim1()?[0])
                            .to_owned()
                            .wrap()
                            .into(),
                        ct,
                    ))
                } else {
                    Ok((
                        arr.select(axis, slc_eval.as_slice().unwrap()).wrap().into(),
                        ct,
                    ))
                }
            },
            RefType::False,
        )
    }

    /// select on a given axis by a slice expression
    ///
    /// # Safety
    ///
    /// The slice should be valid.
    pub unsafe fn select_by_expr_unchecked(self, slc: Expr<'a, usize>, axis: Expr<'a, i32>) -> Self
    where
        T: Clone,
    {
        self.chain_view_f_ct(
            move |(arr, ct)| {
                let (slc, ct) = slc.no_dim0().eval(ct)?;
                let slc_eval = slc.view_arr();
                let axis = *axis.eval(ct.clone())?.0.view_arr().to_dim0()?.into_scalar();
                if slc_eval.ndim() > 1 {
                    return Err("The slice must be dim 0 or dim 1 when select on axis".into());
                }
                let axis = arr.norm_axis(axis);

                if slc_eval.len() == 1 {
                    Ok((
                        arr.index_axis(axis, slc_eval.to_dim1()?[0])
                            .to_owned()
                            .wrap()
                            .into(),
                        ct,
                    ))
                } else {
                    Ok((
                        arr.select_unchecked(axis, slc_eval.as_slice().unwrap())
                            .into(),
                        ct,
                    ))
                }
            },
            RefType::False,
        )
    }

    /// select on a given axis by a slice expression
    pub fn select_by_i32_expr(self, slc: Expr<'a, i32>, axis: Expr<'a, i32>) -> Self
    where
        T: Clone,
    {
        self.chain_view_f_ct(
            move |(arr, ct)| {
                let (slc, ct) = slc.no_dim0().eval(ct)?;
                let slc_eval = slc.view_arr();
                let axis = *axis.eval(ct.clone())?.0.view_arr().to_dim0()?.into_scalar();
                if slc_eval.ndim() > 1 {
                    return Err("The slice must be dim 0 or dim 1 when select on axis".into());
                }
                let axis = arr.norm_axis(axis);
                let length = arr.len_of(axis);
                if slc_eval.len() == 1 {
                    Ok((
                        arr.index_axis(axis, arr.ensure_index(slc_eval.to_dim1()?[0], length))
                            .to_owned()
                            .wrap()
                            .into(),
                        ct,
                    ))
                } else {
                    let slc = slc_eval.to_dim1()?.mapv(|i| arr.ensure_index(i, length));
                    Ok((arr.select(axis, slc.as_slice().unwrap()).wrap().into(), ct))
                }
            },
            RefType::False,
        )
    }

    /// take values on a given axis
    ///
    /// # Safety
    ///
    /// The index in slc must be correct.
    pub unsafe fn take_option_on_axis_by_expr_unchecked(
        self,
        slc: Expr<'a, OptUsize>,
        axis: Expr<'a, i32>,
        par: bool,
    ) -> Self
    where
        T: Clone + GetNone,
    {
        self.chain_view_f_ct(
            move |(arr, ct)| {
                let (slc, ct) = slc.no_dim0().eval(ct)?;
                let slc_eval = slc.view_arr();
                if slc_eval.ndim() > 1 {
                    return Err("The slice must be dim 0 or dim 1 when take on axis".into());
                }
                let axis = *axis.eval(ct.clone())?.0.view_arr().to_dim0()?.into_scalar();
                if slc_eval.len() == 1 {
                    Ok((
                        arr.index_axis(arr.norm_axis(axis), slc_eval.to_dim1()?[0].unwrap())
                            .to_owned()
                            .wrap()
                            .into(),
                        ct,
                    ))
                } else {
                    Ok((
                        arr.take_option_clone_unchecked(slc_eval.to_dim1()?, axis, par)
                            .into(),
                        ct,
                    ))
                }
            },
            RefType::False,
        )
    }

    pub fn where_(self, mask: Expr<'a, bool>, value: Expr<'a, T>, par: bool) -> Self
    where
        T: Clone,
    {
        self.chain_view_f_ct(
            move |(arr, ct)| {
                Ok((
                    arr.where_(
                        &mask.eval(ct.clone())?.0.try_view_arr().unwrap(),
                        &value.eval(ct.clone())?.0.try_view_arr().unwrap(),
                        par,
                    )
                    .into(),
                    ct,
                ))
            },
            RefType::False,
        )
    }

    /// Return the number of dimensions (axes) in the array
    pub fn ndim(self) -> Expr<'a, usize> {
        self.chain_view_f(move |arr| Ok(arr.ndim().into()), RefType::False)
    }

    /// Return the shape of the array as a usize Expr.
    pub fn shape(self) -> Expr<'a, usize> {
        self.chain_view_f(
            move |arr| {
                let shape = arr.shape().to_owned();
                Ok(Arr1::from_vec(shape).to_dimd().into())
            },
            RefType::False,
        )
    }

    pub fn concat(self, other: Vec<Expr<'a, T>>, axis: i32) -> Expr<'a, T>
    where
        T: Clone,
    {
        self.no_dim0().chain_view_f_ct(
            move |(arr, ct)| {
                // evaluate in parallel
                let axis = arr.norm_axis(axis);
                let mut other: Vec<Expr<'a, T>> =
                    other.into_iter().map(|e| e.no_dim0()).collect_trusted();
                other
                    .par_iter_mut()
                    .try_for_each(|e| e.eval_inplace(ct.clone()).map(|_| {}))?;
                let arr1 = vec![arr.0];
                // safety: array view other exists in lifetime '_, and concatenate will create a own array later
                let arrays: Vec<ArrayViewD<'_, T>> = unsafe {
                    arr1.into_iter()
                        .chain(other.iter().map(|e| mem::transmute(e.view().into_arr().0)))
                        .collect()
                };
                Ok((
                    ndarray::concatenate(axis, &arrays)
                        .map_err(|_| "Shape error when concatenate")?
                        // .expect("Shape error when concatenate")
                        .wrap()
                        .into(),
                    ct,
                ))
            },
            RefType::False,
        )
    }

    pub fn stack(self, mut other: Vec<Expr<'a, T>>, axis: i32) -> Expr<'a, T>
    where
        T: Clone,
    {
        self.chain_view_f_ct(
            move |(arr, ct)| {
                // evaluate in parallel
                let axis = if axis < 0 {
                    Axis(arr.norm_axis(axis).index() + 1)
                } else {
                    Axis(axis as usize)
                };
                other
                    .par_iter_mut()
                    .try_for_each(|e| e.eval_inplace(ct.clone()).map(|_| {}))?;
                let arr1 = vec![arr.0];
                // safety: array view other exists in lifetime '_, and stack will create a own array later
                let arrays: Vec<ArrayViewD<'_, T>> = unsafe {
                    arr1.into_iter()
                        .chain(other.iter().map(|e| mem::transmute(e.view().into_arr().0)))
                        .collect()
                };
                Ok((
                    ndarray::stack(axis, &arrays)
                        .map_err(|_| "Shape error when stack")?
                        .wrap()
                        .into(),
                    ct,
                ))
            },
            RefType::False,
        )
    }

    pub fn get_sort_idx(self, mut by: Vec<Exprs<'a>>, rev: bool) -> Expr<'a, usize> {
        self.chain_view_f_ct(
            move |(arr, ct)| {
                if arr.ndim() != 1 {
                    return Err("Currently only 1 dim Expr can be sorted".into());
                }
                let arr = arr.to_dim1()?;
                let len = arr.len();
                // evaluate the key expressions first
                by.par_iter_mut()
                    .try_for_each(|e| e.eval_inplace(ct.clone()).map(|_| {}))?;
                let mut idx = Vec::from_iter(0..len);
                idx.sort_by(move |a, b| {
                    let mut order = Ordering::Equal;
                    for e in by.iter() {
                        let rtn =
                            match &e {
                                Exprs::String(_) | Exprs::DateTime(_) | Exprs::TimeDelta(_) => {
                                    match_exprs!(
                                        e,
                                        e,
                                        {
                                            let key_view = e.view();
                                            let key_arr = key_view.into_arr().to_dim1().expect(
                                                "Currently only 1 dim Expr can be sort key",
                                            );
                                            let (va, vb) =
                                                unsafe { (key_arr.uget(*a), key_arr.uget(*b)) };
                                            if !rev {
                                                va.cmp(vb)
                                            } else {
                                                va.cmp(vb).reverse()
                                            }
                                        },
                                        String,
                                        DateTime
                                    )
                                }
                                _ => {
                                    match_exprs!(
                                        e,
                                        e,
                                        {
                                            let key_view = e.view();
                                            let key_arr = key_view.into_arr().to_dim1().expect(
                                                "Currently only 1 dim Expr can be sort key",
                                            );
                                            let (va, vb) =
                                                unsafe { (key_arr.uget(*a), key_arr.uget(*b)) };
                                            if !rev {
                                                va.nan_sort_cmp_stable(vb)
                                            } else {
                                                va.nan_sort_cmp_rev_stable(vb)
                                            }
                                        },
                                        F64,
                                        F32,
                                        I64,
                                        I32,
                                        Usize,
                                        #[cfg(feature = "option_dtype")]
                                        OptF64,
                                        #[cfg(feature = "option_dtype")]
                                        OptF32,
                                        #[cfg(feature = "option_dtype")]
                                        OptI64,
                                        #[cfg(feature = "option_dtype")]
                                        OptI32
                                    )
                                }
                            };
                        if rtn != Ordering::Equal {
                            order = rtn;
                            break;
                        }
                    }
                    order
                });
                Ok((Arr1::from_vec(idx).to_dimd().into(), ct))
            },
            RefType::False,
        )
    }

    pub fn sort_by_expr(self, by: Vec<Exprs<'a>>, rev: bool) -> Self
    where
        T: Clone,
    {
        let idx = self.clone().get_sort_idx(by, rev);
        // safety: the idx is valid
        unsafe { self.select_by_expr_unchecked(idx, 0.into()) }
    }

    // pub fn apply_on_vec_arr<F>(self, func: F, axis: i32) -> Expr<'a, T>
    // {
    //     self.chain_f(|input| {
    //         if let ExprOut::ArrVec(arr_vec) = input {
    //             let view_vec = arr_vec.iter().map(|arr| arr.view()).collect_trusted();
    //             use ndarray::Zip;
    //             let shape =
    //             let out = Vec::with_capacity(arr_vec.len());
    //         }

    //     })
    // }
}

impl<'a> Expr<'a, bool> {
    pub fn any(self, axis: i32, par: bool) -> Self {
        self.chain_view_f(move |arr| Ok(arr.any(axis, par).into()), RefType::False)
    }

    pub fn all(self, axis: i32, par: bool) -> Self {
        self.chain_view_f(move |arr| Ok(arr.all(axis, par).into()), RefType::False)
    }
}

impl<'a> Expr<'a, PyValue> {
    pub fn object_to_string(self, py: Python) -> TpResult<Expr<'a, String>> {
        let e = self.eval(None)?.0;
        let name = e.name();
        let out = e.view_arr().object_to_string(py);
        Ok(Expr::new(out.into(), name))
    }

    // pub fn object_to_str<'py: 'a>(self, py: Python<'py>) -> Expr<'a, &'py str>
    // {
    //     let e = self.eval();
    //     let name = e.name();
    //     let out = e.view_arr().object_to_str(py);
    //     Expr::<'a, &'py str>::new(out.into(), name)
    // }
}

impl<'a> Expr<'a, String> {
    pub fn strptime<'b: 'a>(self, fmt: String) -> Expr<'a, DateTime> {
        self.chain_view_f(
            move |arr| {
                Ok(arr
                    .map(move |s| DateTime::parse(s, &fmt).unwrap_or_default())
                    .into())
            },
            RefType::False,
        )
    }

    pub fn add_str(self, other: Expr<'a, &'a str>) -> Expr<'a, String> {
        self.chain_view_f_ct(
            |(arr, ct)| {
                let (other, ct) = other.eval(ct)?;
                Ok((
                    Zip::from(arr.0)
                        .and(other.view_arr().0)
                        .par_map_collect(|s1, s2| s1.to_owned() + s2)
                        .wrap()
                        .into(),
                    ct,
                ))
            },
            RefType::False,
        )
    }

    pub fn add_string(self, other: Expr<'a, String>) -> Expr<'a, String> {
        self.chain_view_f_ct(
            |(arr, ct)| {
                let (other, ct) = other.eval(ct)?;
                Ok((
                    Zip::from(arr.0)
                        .and(other.view_arr().0)
                        .par_map_collect(|s1, s2| s1.to_owned() + s2)
                        .wrap()
                        .into(),
                    ct,
                ))
            },
            RefType::False,
        )
    }
}

impl<'a> Expr<'a, DateTime> {
    pub fn strftime(self, fmt: Option<String>) -> Expr<'a, String> {
        self.chain_view_f(
            move |arr| Ok(arr.map(move |dt| dt.strftime(fmt.clone())).into()),
            RefType::False,
        )
    }
}
