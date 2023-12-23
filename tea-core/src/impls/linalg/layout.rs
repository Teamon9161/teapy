use crate::{
    prelude::{Arr2, ArrBase, TpResult, WrapNdarray},
    utils::{vec_uninit, VecAssumeInit},
};
use error::StrError;
use ndarray::{ArrayBase, Data, Dimension};
use std::mem::MaybeUninit;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    C { row: i32, lda: i32 },
    F { col: i32, lda: i32 },
}

impl MatrixLayout {
    #[inline(always)]
    pub fn size(&self) -> (i32, i32) {
        match *self {
            MatrixLayout::C { row, lda } => (row, lda),
            MatrixLayout::F { col, lda } => (lda, col),
        }
    }

    #[inline(always)]
    pub fn resized(&self, row: i32, col: i32) -> MatrixLayout {
        match *self {
            MatrixLayout::C { .. } => MatrixLayout::C { row, lda: col },
            MatrixLayout::F { .. } => MatrixLayout::F { col, lda: row },
        }
    }

    #[inline(always)]
    pub fn lda(&self) -> i32 {
        std::cmp::max(
            1,
            match *self {
                MatrixLayout::C { lda, .. } | MatrixLayout::F { lda, .. } => lda,
            },
        )
    }

    #[inline(always)]
    pub fn len(&self) -> i32 {
        match *self {
            MatrixLayout::C { row, .. } => row,
            MatrixLayout::F { col, .. } => col,
        }
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn same_order(&self, other: &MatrixLayout) -> bool {
        matches!(
            (self, other),
            (MatrixLayout::C { .. }, MatrixLayout::C { .. })
                | (MatrixLayout::F { .. }, MatrixLayout::F { .. })
        )
    }

    #[inline(always)]
    pub fn toggle_order(&self) -> Self {
        match *self {
            MatrixLayout::C { row, lda } => MatrixLayout::F { lda: row, col: lda },
            MatrixLayout::F { col, lda } => MatrixLayout::C { row: lda, lda: col },
        }
    }

    /// Transpose without changing memory representation
    ///
    /// C-contigious row=2, lda=3
    ///
    /// ```text
    /// [[1, 2, 3]
    ///  [4, 5, 6]]
    /// ```
    ///
    /// and F-contigious col=2, lda=3
    ///
    /// ```text
    /// [[1, 4]
    ///  [2, 5]
    ///  [3, 6]]
    /// ```
    ///
    /// have same memory representation `[1, 2, 3, 4, 5, 6]`, and this toggles them.
    ///
    /// ```
    /// # use lax::layout::*;
    /// let layout = MatrixLayout::C { row: 2, lda: 3 };
    /// assert_eq!(layout.t(), MatrixLayout::F { col: 2, lda: 3 });
    /// ```
    #[inline(always)]
    pub fn t(&self) -> Self {
        match *self {
            MatrixLayout::C { row, lda } => MatrixLayout::F { col: row, lda },
            MatrixLayout::F { col, lda } => MatrixLayout::C { row: col, lda },
        }
    }
}

/// Out-place transpose for general matrix
///
/// Examples
/// ---------
///
/// ```rust
/// # use lax::layout::*;
/// let layout = MatrixLayout::C { row: 2, lda: 3 };
/// let a = vec![1., 2., 3., 4., 5., 6.];
/// let (l, b) = transpose(layout, &a);
/// assert_eq!(l, MatrixLayout::F { col: 3, lda: 2 });
/// assert_eq!(b, &[1., 4., 2., 5., 3., 6.]);
/// ```
///
/// ```rust
/// # use lax::layout::*;
/// let layout = MatrixLayout::F { col: 2, lda: 3 };
/// let a = vec![1., 2., 3., 4., 5., 6.];
/// let (l, b) = transpose(layout, &a);
/// assert_eq!(l, MatrixLayout::C { row: 3, lda: 2 });
/// assert_eq!(b, &[1., 4., 2., 5., 3., 6.]);
/// ```
///
/// Panics
/// ------
/// - If input array size and `layout` size mismatch
///
pub fn transpose<T: Copy>(layout: MatrixLayout, input: &[T]) -> (MatrixLayout, Vec<T>) {
    let (m, n) = layout.size();
    let transposed = layout.resized(n, m).t();
    let m = m as usize;
    let n = n as usize;
    assert_eq!(input.len(), m * n);

    let mut out: Vec<MaybeUninit<T>> = vec_uninit(m * n);

    match layout {
        MatrixLayout::C { .. } => {
            for i in 0..m {
                for j in 0..n {
                    out[j * m + i].write(input[i * n + j]);
                }
            }
        }
        MatrixLayout::F { .. } => {
            for i in 0..m {
                for j in 0..n {
                    out[i * n + j].write(input[j * m + i]);
                }
            }
        }
    }
    (transposed, unsafe { out.assume_init() })
}

/// Out-place transpose for general matrix
///
/// Examples
/// ---------
///
/// ```rust
/// # use lax::layout::*;
/// let layout = MatrixLayout::C { row: 2, lda: 3 };
/// let a = vec![1., 2., 3., 4., 5., 6.];
/// let mut b = vec![0.0; a.len()];
/// let l = transpose_over(layout, &a, &mut b);
/// assert_eq!(l, MatrixLayout::F { col: 3, lda: 2 });
/// assert_eq!(b, &[1., 4., 2., 5., 3., 6.]);
/// ```
///
/// ```rust
/// # use lax::layout::*;
/// let layout = MatrixLayout::F { col: 2, lda: 3 };
/// let a = vec![1., 2., 3., 4., 5., 6.];
/// let mut b = vec![0.0; a.len()];
/// let l = transpose_over(layout, &a, &mut b);
/// assert_eq!(l, MatrixLayout::C { row: 3, lda: 2 });
/// assert_eq!(b, &[1., 4., 2., 5., 3., 6.]);
/// ```
///
/// Panics
/// ------
/// - If input array sizes and `layout` size mismatch
///
pub fn transpose_over<T: Copy>(layout: MatrixLayout, from: &[T], to: &mut [T]) -> MatrixLayout {
    let (m, n) = layout.size();
    let transposed = layout.resized(n, m).t();
    let m = m as usize;
    let n = n as usize;
    assert_eq!(from.len(), m * n);
    assert_eq!(to.len(), m * n);

    match layout {
        MatrixLayout::C { .. } => {
            for i in 0..m {
                for j in 0..n {
                    to[j * m + i] = from[i * n + j];
                }
            }
        }
        MatrixLayout::F { .. } => {
            for i in 0..m {
                for j in 0..n {
                    to[i * n + j] = from[j * m + i];
                }
            }
        }
    }
    transposed
}

pub fn into_matrix<T>(l: MatrixLayout, a: Vec<T>) -> TpResult<Arr2<T>> {
    use ndarray::ShapeBuilder;
    match l {
        MatrixLayout::C { row, lda } => {
            Ok(ArrayBase::from_shape_vec((row as usize, lda as usize), a)
                .map_err(|e| StrError::from(format!("{e:?}")))?
                .wrap())
        }
        MatrixLayout::F { col, lda } => Ok(ArrayBase::from_shape_vec(
            (lda as usize, col as usize).f(),
            a,
        )
        .map_err(|e| StrError::from(format!("{e:?}")))?
        .wrap()),
    }
}

impl<T, S, D> ArrBase<S, D>
where
    S: Data<Elem = T>,
    D: Dimension,
{
    pub fn layout(&self) -> TpResult<MatrixLayout> {
        let shape = self.shape();
        let strides = self.strides();
        match self.ndim() {
            1 => Ok(MatrixLayout::F {
                col: self.len() as i32,
                lda: 1,
            }),
            2 => {
                let arr2d: &Arr2<T> = unsafe { std::mem::transmute(self) };
                if shape[0] == strides[1] as usize {
                    return Ok(MatrixLayout::F {
                        col: arr2d.ncols() as i32,
                        lda: arr2d.nrows() as i32,
                    });
                }
                if shape[1] == strides[0] as usize {
                    return Ok(MatrixLayout::C {
                        row: arr2d.nrows() as i32,
                        lda: arr2d.ncols() as i32,
                    });
                }
                Err("Invalid stride of ndim2".into())
            }
            _ => Err("Invalid dimension".into()),
        }
    }
}
