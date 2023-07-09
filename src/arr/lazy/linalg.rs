use super::{ArbArray, Expr, ExprElement, ExprInner, RefType};
use crate::arr::{Arr1, Arr2, ArrD, ArrViewD, Axis, WrapNdarray};
use ndarray::Ix2;
use ndarray::LinalgScalar;
// use ndarray_linalg::{conjugate};
use super::super::impls::{conjugate, LeastSquaresResult};
use num::traits::AsPrimitive;
use std::sync::Arc;

// we created a new struct for linear square result as we need a result
// with only one generic
#[derive(Default, Debug)]
pub struct OlsResult<'a> {
    pub x: ArbArray<'a, f64>,
    pub y: Expr<'a, f64>,
    /// The singular values of the matrix A in `Ax = b`
    pub singular_values: Arr1<f64>,
    /// The solution vector or matrix `x` which is the best
    /// solution to `Ax = b`, i.e. minimizing the 2-norm `||b - Ax||`
    pub solution: ArrD<f64>,
    /// The rank of the matrix A in `Ax = b`
    pub rank: i32,
    /// If n < m and rank(A) == n, the sum of squares
    /// If b is a (m x 1) vector, this is a 0-dimensional array (single value)
    /// If b is a (m x k) matrix, this is a (k x 1) column vector
    pub residual_sum_of_squares: Option<ArrD<f64>>,
}

impl<'a> OlsResult<'a> {
    // pub fn new<D: Dimension>(
    //     lstsq_result: LeastSquaresResult<f64, D>,
    //     x: ArbArray<'a, f64>,
    //     y: Expr<'a, f64>,
    // ) -> Self {
    //     OlsResult {
    //         x,
    //         y,
    //         singular_values: lstsq_result.singular_values.wrap(),
    //         solution: lstsq_result.solution.wrap().to_dimd().unwrap(),
    //         rank: lstsq_result.rank,
    //         residual_sum_of_squares: lstsq_result
    //             .residual_sum_of_squares
    //             .map(|arr| arr.wrap().to_dimd().unwrap()),
    //     }
    // }
    pub fn new(lstsq_result: LeastSquaresResult, x: ArbArray<'a, f64>, y: Expr<'a, f64>) -> Self {
        OlsResult {
            x,
            y,
            singular_values: lstsq_result.singular_values,
            solution: lstsq_result.solution.unwrap(),
            rank: lstsq_result.rank,
            residual_sum_of_squares: lstsq_result.residual_sum_of_squares,
        }
    }

    pub fn x(&self) -> ArrViewD<'_, f64> {
        self.x.view()
    }

    pub fn y(&self) -> ArrViewD<'_, f64> {
        self.y.view_arr()
    }

    pub fn singular_values(&self) -> ArrViewD<'_, f64> {
        self.singular_values.view().to_dimd().unwrap()
    }

    pub fn solution(&self) -> ArrViewD<'_, f64> {
        self.solution.view()
    }

    pub fn rank(&self) -> i32 {
        self.rank
    }

    pub fn residual_sum_of_squares(&self) -> ArrViewD<'_, f64> {
        self.residual_sum_of_squares.as_ref().unwrap().view()
    }
}

impl<'a, T: ExprElement> ExprInner<'a, T> {
    pub fn chain_ols_f<T2, F>(self, f: F) -> ExprInner<'a, T2>
    where
        F: FnOnce(Arc<OlsResult<'_>>) -> ArbArray<'a, T2> + Send + Sync + 'a,
        T2: ExprElement,
    {
        self.chain_f(
            |expr_out| f(expr_out.into_ols_result()).into(),
            RefType::False,
        )
        // ExprInner::<'a, T2> {
        //     base: std::mem::take(&mut self.base),
        //     step: self.step + 1,
        //     name: self.name,
        //     owned: None,
        //     func: Box::new(move |base: ExprBase<'a>| {
        //         let res = (self.func)(base).into_ols_result();
        //         f(res).into()
        //     }),
        //     ref_expr: None,
        // }
    }
}

impl<'a, T: ExprElement + 'a> Expr<'a, T> {
    pub fn chain_ols_f<T2, F>(self, f: F) -> Expr<'a, T2>
    where
        F: FnOnce(Arc<OlsResult<'_>>) -> ArbArray<'a, T2> + Send + Sync + 'a,
        T2: ExprElement + 'a,
    {
        self.downcast().chain_ols_f(f).into()
    }

    pub fn lstsq<T2: ExprElement + AsPrimitive<f64>>(self, y: Expr<'a, T2>) -> Self
    where
        T: AsPrimitive<f64>,
    {
        self.cast::<f64>().chain_f(
            move |x| {
                let x = x.into_arr();
                let y = y.cast::<f64>().eval();
                let y_arr = y.view_arr();
                let x_view = match x.ndim() {
                    1 => x.view().to_dim1().unwrap().insert_axis(Axis(1)).wrap(),
                    2 => x.view().to_dim::<Ix2>().unwrap(),
                    _ => panic!("Too much dimension in lstsq"),
                };
                OlsResult::new(
                    x_view
                        .to_owned()
                        .least_squares(&mut y_arr.to_owned())
                        .unwrap(),
                    x,
                    y,
                )
                .into()
            },
            RefType::False,
        )
    }

    pub fn ols_rank(self) -> Expr<'a, i32> {
        self.chain_ols_f(|res| res.rank().into())
    }

    pub fn singular_values(self) -> Expr<'a, f64> {
        self.chain_ols_f(|res| match Arc::try_unwrap(res) {
            Ok(res) => res.singular_values.to_dimd().unwrap().into(),
            Err(res) => res.singular_values().to_owned().into(),
        })
    }

    pub fn params(self) -> Expr<'a, f64> {
        self.chain_ols_f(|res| match Arc::try_unwrap(res) {
            Ok(res) => res.solution.to_dimd().unwrap().into(),
            Err(res) => res.solution().to_owned().into(),
        })
    }

    pub fn sse(self) -> Expr<'a, f64> {
        self.chain_ols_f(|res| match Arc::try_unwrap(res) {
            Ok(res) => res
                .residual_sum_of_squares
                .unwrap()
                .to_dimd()
                .unwrap()
                .into(),
            Err(res) => res.residual_sum_of_squares().to_owned().into(),
        })
    }

    pub fn fitted_values(self) -> Expr<'a, f64> {
        self.chain_ols_f(|res| match Arc::try_unwrap(res) {
            Ok(res) => res.x.view().dot(&res.solution).into(),
            Err(res) => {
                let x = res.x();
                let params = res.solution();
                x.dot(&params).into()
            }
        })
    }

    pub fn dot(self, other: Expr<'a, T>) -> Expr<'a, T>
    where
        T: LinalgScalar,
    {
        self.chain_view_f(
            move |arr| {
                let other = other.eval();
                arr.dot(&other.view_arr()).into()
            },
            RefType::False,
        )
    }

    pub fn conjugate(self) -> Expr<'a, f64>
    where
        T: AsPrimitive<f64>,
    {
        self.cast::<f64>().chain_view_f(
            move |arr| {
                let arr = arr.to_dim2().expect("Array should be dim2 when conjugate");
                let out: Arr2<f64> = conjugate(&arr);
                out.to_dimd().unwrap().into()
            },
            RefType::False,
        )
    }

    pub fn svd(self, full: bool, calc_uvt: bool) -> Expr<'a, f64>
    where
        T: AsPrimitive<f64>,
    {
        self.cast::<f64>().chain_f(
            move |arr| {
                let arr = arr.into_arr().try_to_owned_f();
                let (u, s, vt) = arr.svd_into(full, calc_uvt);
                if !calc_uvt {
                    s.into()
                } else {
                    let res_vec: Vec<ArbArray<'a, f64>> =
                        vec![u.unwrap().into(), s.into(), vt.unwrap().into()];
                    res_vec.into()
                }
            },
            RefType::False,
        )
    }

    pub fn pinv(self, rcond: Option<f64>, return_s: bool) -> Expr<'a, f64>
    where
        T: AsPrimitive<f64>,
    {
        self.cast::<f64>().chain_f(
            move |arr| {
                // we don't need to conjugate because the dtype is float64
                let arr = arr.into_arr().try_to_owned_f();
                let shape = arr.shape();
                let m = shape[0];
                let n = shape[1];
                let (u, s, vt) = arr.svd_into(false, true);
                let u = u.unwrap().to_dim2().unwrap();
                let vt = vt.unwrap().to_dim2().unwrap();
                let s = s.to_dim1().unwrap();
                let rcond = rcond.unwrap_or(f64::EPSILON * m.max(n) as f64);
                let cutoff = rcond * s.max_1d();
                let s = s.mapv(|v| if v > cutoff { 1. / v } else { 0. });
                let out: ArbArray<'a, f64> = vt
                    .t()
                    .dot(&(s.to_owned().0.insert_axis(Axis(1)) * u.t()))
                    .wrap()
                    .to_dimd()
                    .unwrap()
                    .into();
                if return_s {
                    vec![out, s.to_dimd().unwrap().into()].into()
                } else {
                    out.into()
                }
            },
            RefType::False,
        )
    }
}
