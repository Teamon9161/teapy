use super::super::super::{
    utils::{vec_uninit, VecAssumeInit},
    Arr, Arr1, Arr2, ArrD, WrapNdarray,
};
use super::{transpose, transpose_over, MatrixLayout};
use lapack_sys::dgelsd_;
use ndarray::{s, Axis, Ix0, ShapeBuilder};
use std::mem::transmute;

#[derive(Debug, Clone)]
pub struct LeastSquaresResult {
    /// The singular values of the matrix A in `Ax = b`
    pub singular_values: Arr1<f64>,
    /// The solution vector or matrix `x` which is the best
    /// solution to `Ax = b`, i.e. minimizing the 2-norm `||b - Ax||`
    pub solution: Option<ArrD<f64>>,
    /// The rank of the matrix A in `Ax = b`
    pub rank: i32,
    /// If n < m and rank(A) == n, the sum of squares
    /// If b is a (m x 1) vector, this is a 0-dimensional array (single value)
    /// If b is a (m x k) matrix, this is a (k x 1) column vector
    pub residual_sum_of_squares: Option<ArrD<f64>>,
}

impl Arr2<f64> {
    pub fn least_squares(
        &mut self,
        rhs: &mut ArrD<f64>,
    ) -> Result<LeastSquaresResult, &'static str> {
        if self.shape()[0] != rhs.shape()[0] {
            return Err("Invalid shape in least squares");
        }
        match rhs.ndim() {
            1 => {
                let (m, n) = (self.shape()[0], self.shape()[1]);
                let mut res = if n > m {
                    // we need a new rhs b/c it will be overwritten with the solution
                    // for which we need `n` entries
                    let mut new_rhs = Arr1::<f64>::zeros((n,));
                    new_rhs.slice_mut(s![0..m]).assign(rhs);
                    let mut new_rhs = new_rhs.to_dimd().unwrap();
                    let a_layout = self.layout()?;
                    let b_layout = new_rhs.layout()?;
                    least_squares_impl(self, a_layout, &mut new_rhs, b_layout)?
                } else {
                    let a_layout = self.layout()?;
                    let b_layout = rhs.layout()?;
                    least_squares_impl(self, a_layout, rhs, b_layout)?
                };
                res.solution = Some(rhs.slice(s![0..n]).to_owned().wrap().to_dimd().unwrap());
                res.residual_sum_of_squares =
                    compute_residual_scalar(m, n, res.rank, unsafe { transmute(rhs) })
                        .map(|r| r.to_dimd().unwrap());
                Ok(res)
            }
            2 => {
                let (m, n) = (self.shape()[0], self.shape()[1]);
                let mut res = if n > m {
                    // we need a new rhs b/c it will be overwritten with the solution
                    // for which we need `n` entries
                    let k = rhs.shape()[1];
                    let mut new_rhs = match self.layout()? {
                        MatrixLayout::C { .. } => Arr2::<f64>::zeros((n, k)),
                        MatrixLayout::F { .. } => Arr2::<f64>::zeros((n, k).f()),
                    };
                    new_rhs.slice_mut(s![0..m, ..]).assign(rhs);
                    let mut new_rhs = new_rhs.to_dimd().unwrap();
                    let a_layout = self.layout()?;
                    let b_layout = new_rhs.layout()?;
                    least_squares_impl(self, a_layout, &mut new_rhs, b_layout)?
                } else {
                    let a_layout = self.layout()?;
                    let b_layout = rhs.layout()?;
                    least_squares_impl(self, a_layout, rhs, b_layout)?
                };
                res.solution = Some(rhs.slice(s![0..n]).to_owned().wrap().to_dimd().unwrap());
                res.residual_sum_of_squares =
                    compute_residual_array1(m, n, res.rank, unsafe { transmute(rhs) })
                        .map(|r| r.to_dimd().unwrap());
                Ok(res)
            }
            _ => Err("Invalid dimension in least squares"),
        }
    }
}

fn compute_residual_scalar(m: usize, n: usize, rank: i32, b: &Arr1<f64>) -> Option<Arr<f64, Ix0>> {
    if m < n || n != rank as usize {
        return None;
    }
    let mut arr: Arr<f64, Ix0> = Arr::zeros(());
    arr[()] = b.slice(s![n..]).mapv(|x| x.powi(2).abs()).sum();
    Some(arr)
}

fn compute_residual_array1(m: usize, n: usize, rank: i32, b: &Arr2<f64>) -> Option<Arr1<f64>> {
    if m < n || n != rank as usize {
        return None;
    }
    Some(
        b.slice(s![n.., ..])
            .mapv(|x| x.powi(2).abs())
            .sum_axis(Axis(0))
            .wrap(),
    )
}

pub fn least_squares_impl(
    a: &mut Arr2<f64>,
    a_layout: MatrixLayout,
    b: &mut ArrD<f64>,
    b_layout: MatrixLayout,
) -> Result<LeastSquaresResult, &'static str> {
    let mut_a = a
        .as_slice_memory_order_mut()
        .expect("Array should be contiguous when lstsq");
    let mut_b = b
        .as_slice_memory_order_mut()
        .expect("Array should be contiguous when lstsq");
    let (m, n) = a_layout.size();
    let (m_, nrhs) = b_layout.size();
    let k = m.min(n);
    assert!(m_ >= m);
    let rcond = -1.;
    let mut singular_values = vec_uninit::<f64>(k as usize);
    let mut rank: i32 = 0;
    // eval work size
    let mut info = 0;
    let mut work_size = [0.];
    let mut iwork_size = [0];
    unsafe {
        dgelsd_(
            &m,
            &n,
            &nrhs,
            std::ptr::null_mut(),
            &m,
            std::ptr::null_mut(),
            &m_,
            singular_values.as_mut_ptr() as *mut f64,
            &rcond,
            &mut rank,
            work_size.as_mut_ptr(),
            &(-1),
            iwork_size.as_mut_ptr(),
            &mut info,
        )
    }
    if info != 0 {
        panic!("lstsq error: info = {info}");
    }
    let lwork = work_size[0] as i32;
    let mut work = vec_uninit::<f64>(lwork.try_into().unwrap());
    let liwork = iwork_size[0];
    let mut iwork = vec_uninit::<i32>(liwork.try_into().unwrap());
    info = 0;
    rank = 0;
    // Transpose if a is C-continuous
    let mut a_t = None;
    let _ = match a_layout {
        MatrixLayout::C { .. } => {
            let (layout, t) = transpose(a_layout, mut_a);
            a_t = Some(t);
            layout
        }
        MatrixLayout::F { .. } => a_layout,
    };

    // Transpose if b is C-continuous
    let mut b_t = None;
    let b_layout = match b_layout {
        MatrixLayout::C { .. } => {
            let (layout, t) = transpose(b_layout, mut_b);
            b_t = Some(t);
            layout
        }
        MatrixLayout::F { .. } => b_layout,
    };
    unsafe {
        dgelsd_(
            &m,
            &n,
            &nrhs,
            a_t.as_mut()
                .map(|x| x.as_mut_ptr())
                .unwrap_or(mut_a.as_mut_ptr()) as *mut f64,
            &m,
            b_t.as_mut()
                .map(|x| x.as_mut_ptr())
                .unwrap_or(mut_b.as_mut_ptr()) as *mut f64,
            &m_,
            singular_values.as_mut_ptr() as *mut f64,
            &rcond,
            &mut rank,
            work.as_mut_ptr() as *mut f64,
            &lwork,
            iwork.as_mut_ptr() as *mut i32,
            &mut info,
        );
    }
    if info != 0 {
        panic!("lstsq error: info = {info}");
    }
    // Skip a_t -> a transpose because A has been destroyed
    // Re-transpose b
    if let Some(b_t) = b_t {
        transpose_over(b_layout, &b_t, mut_b);
    }

    Ok(LeastSquaresResult {
        singular_values: unsafe { Arr1::from_vec(singular_values.assume_init()) },
        solution: None,
        rank,
        residual_sum_of_squares: None,
    })
}
