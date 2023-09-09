use super::super::core::{conjugate, LeastSquaresResult};
use super::expr_core::Expr;
use crate::{ArbArray, Arr1, Arr2, ArrD, ArrOk, ArrViewD, WrapNdarray};
use ndarray::{Array1, Axis, Ix2, Slice};
use std::sync::Arc;

// we created a new struct for linear square result as we need a result
// with only one generic
#[derive(Default, Debug)]
pub struct OlsResult<'a> {
    pub x: ArbArray<'a, f64>,
    pub y: ArbArray<'a, f64>,
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
    pub fn new(
        lstsq_result: LeastSquaresResult,
        x: ArbArray<'a, f64>,
        y: ArbArray<'a, f64>,
    ) -> Self {
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
        self.y.view()
    }

    pub fn singular_values(&self) -> ArrViewD<'_, f64> {
        self.singular_values.view().to_dimd()
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

impl<'a> Expr<'a> {
    pub fn lstsq(&mut self, y: Expr<'a>) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let x = data.into_arr(ctx.clone())?.cast_f64();
            let y = y.clone();
            let y = y.into_arr(ctx.clone())?.cast_f64();
            let y_view = y.view();
            let x_view = match x.ndim() {
                1 => x.view().to_dim1().unwrap().insert_axis(Axis(1)).wrap(),
                2 => x.view().to_dim::<Ix2>().unwrap(),
                _ => return Err("Too much dimension in lstsq".into()),
            };
            Ok((
                OlsResult::new(
                    x_view
                        .to_owned()
                        .least_squares(&mut y_view.to_owned())
                        .unwrap(),
                    x,
                    y,
                )
                .into(),
                ctx,
            ))
        });
        self
    }

    pub fn ols_rank(&mut self) -> &mut Self {
        self.chain_f(|data| {
            let res = data.into_ols_res()?;
            Ok(res.rank().into())
        });
        self
    }

    pub fn singular_values(&mut self) -> &mut Self {
        self.chain_f(|data| {
            let res = data.into_ols_res()?;
            match Arc::try_unwrap(res) {
                Ok(res) => Ok(res.singular_values.to_dimd().into()),
                Err(res) => Ok(res.singular_values().to_owned().into()),
            }
        });
        self
    }

    pub fn params(&mut self) -> &mut Self {
        self.chain_f(|data| {
            let res = data.into_ols_res()?;
            match Arc::try_unwrap(res) {
                Ok(res) => Ok(res.solution.to_dimd().into()),
                Err(res) => Ok(res.solution().to_owned().into()),
            }
        });
        self
    }

    pub fn sse(&mut self) -> &mut Self {
        self.chain_f(|data| {
            let res = data.into_ols_res()?;
            match Arc::try_unwrap(res) {
                Ok(res) => Ok(res.residual_sum_of_squares.unwrap().to_dimd().into()),
                Err(res) => Ok(res.residual_sum_of_squares().to_owned().into()),
            }
        });
        self
    }

    pub fn fitted_values(&mut self) -> &mut Self {
        self.chain_f(|data| {
            let res = data.into_ols_res()?;
            match Arc::try_unwrap(res) {
                Ok(res) => Ok(res.x.view().dot(&res.solution)?.into()),
                Err(res) => {
                    let x = res.x();
                    let params = res.solution();
                    Ok(x.dot(&params)?.into())
                }
            }
        });
        self
    }

    // pub fn dot1(&mut self, other: Expr<'a>) -> &mut Self
    // {
    //     self.chain_f_ctx(
    //         move |(data, ctx)| {
    //             let arr = data.into_arr(ctx.clone())?.cast_f64();
    //             let other = other.into_arr(ctx.clone())?.cast_f64();
    //             Ok((arr.view().dot(&other.view())?.into(), ctx))
    //         },
    //     );
    //     self
    // }

    pub fn conjugate(&mut self) -> &mut Self {
        self.chain_f(move |data| {
            let arr = data.into_arr(None)?.cast_f64();
            let out: Arr2<f64> = conjugate(&arr.view().to_dim2()?);
            Ok(out.to_dimd().into())
        });
        self
    }

    pub fn svd(&mut self, full: bool, calc_uvt: bool) -> &mut Self {
        self.chain_f(move |data| {
            let arr = data.into_arr(None)?.cast_f64().try_to_owned_f();
            let (u, s, vt) = arr.svd_into(full, calc_uvt)?;
            if !calc_uvt {
                Ok(s.into())
            } else {
                let res_vec: Vec<ArrOk<'a>> = vec![u.unwrap().into(), s.into(), vt.unwrap().into()];
                Ok(res_vec.into())
            }
        });
        self
    }

    pub fn pinv(&mut self, rcond: Option<f64>, return_s: bool) -> &mut Self {
        self.chain_f(move |arr| {
            // we don't need to conjugate because the dtype is float64
            let arr = arr.into_arr(None)?.cast_f64().try_to_owned_f();
            let shape = arr.shape();
            let m = shape[0];
            let n = shape[1];
            let (u, s, vt) = arr.svd_into(false, true)?;
            let u = u.unwrap().to_dim2().unwrap();
            let vt = vt.unwrap().to_dim2().unwrap();
            let s = s.to_dim1().unwrap();
            let rcond = rcond.unwrap_or(f64::EPSILON * m.max(n) as f64);
            let cutoff = rcond * s.max_1d();
            let s = s.mapv(|v| if v > cutoff { 1. / v } else { 0. });
            let out: ArrOk<'a> = vt
                .t()
                .dot(&(s.to_owned().0.insert_axis(Axis(1)) * u.t()))
                .wrap()
                .to_dimd()
                .into();
            if return_s {
                Ok(vec![out, s.to_dimd().into()].into())
            } else {
                Ok(out.into())
            }
        });
        self
    }

    pub fn get_newey_west_adjust_s(&mut self, resid: Expr<'a>, lag: Expr<'a>) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let x = data.into_arr(ctx.clone())?.cast_f64().into_owned();
            let resid = resid.view_arr(ctx.as_ref())?.deref().cast_f64();
            let resid_view = resid.view().to_dim1()?;
            let lag = lag
                .view_arr(ctx.as_ref())?
                .deref()
                .cast_usize()
                .into_owned()
                .to_dim0()?
                .into_scalar();
            let lag_f64: f64 = lag as f64;
            let weights = Array1::range(0., lag_f64 + 1., 1.)
                .mapv(|v| 1. - v / (lag_f64 + 1.))
                .wrap();
            let score = x.0 * resid_view.insert_axis(Axis(1));
            let mut s = score.t().wrap().dot(&score.view().wrap())?.0;
            for lag in 1..=lag {
                let temp = score
                    .slice_axis(Axis(0), Slice::new(lag as isize, None, 1))
                    .t()
                    .wrap()
                    .dot(
                        &score
                            .slice_axis(Axis(0), Slice::new(0, Some(-(lag as isize)), 1))
                            .wrap(),
                    )?
                    .0;
                s = s + *weights.get(lag).unwrap() * (temp.to_owned() + temp.t());
            }
            let out: ArrOk<'a> = s.wrap().to_dimd().into();
            Ok((out.into(), ctx))
        });
        self
    }
}
