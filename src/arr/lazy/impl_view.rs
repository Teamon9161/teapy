//! impl methods that may return an array view.
//! these method are unsafe because the data of the array
//! view should not be dropped.
//! the memory should be managed on python heap if using in python.

use super::super::WrapNdarray;
use super::{ArbArray, Expr, ExprElement, RefType};
use ndarray::{s, IxDyn, NewAxis, SliceInfo, SliceInfoElem};
use std::marker::PhantomData;
use std::mem;

impl<'a, T> Expr<'a, T>
where
    T: ExprElement + 'a,
{
    /// # Safety
    ///
    /// the data for the array view should exist
    pub unsafe fn reshape(self, shape: Expr<'a, usize>) -> Self {
        self.chain_view_f(
            |arr| {
                let shape = shape.eval();
                let sh = shape.view_arr();
                let ndim = sh.ndim();
                if ndim == 0 {
                    let shape = sh.to_dim0().unwrap().into_scalar();
                    let out: ArbArray<'_, T> = arr
                        .0
                        .into_shape(*shape)
                        .expect("Shape Error")
                        .wrap()
                        .to_dimd()
                        .unwrap()
                        .into();
                    // safe because the view exist in lifetime 'a
                    mem::transmute(out)
                } else if ndim == 1 {
                    let shape = sh.to_dim1().unwrap();
                    let out: ArbArray<'_, T> = arr
                        .0
                        .into_shape(shape.to_slice().unwrap())
                        .expect("Shape Error")
                        .wrap()
                        .to_dimd()
                        .unwrap()
                        .into();
                    mem::transmute(out)
                } else {
                    panic!("the dim of shape should not be greater than 1")
                }
            },
            RefType::True,
        )
    }

    /// convert dim0 output to dim1
    pub fn no_dim0(self) -> Self
    where
        T: Clone,
    {
        self.chain_arr_f(
            move |arr| {
                if arr.ndim() == 0 {
                    arr.to_owned()
                        .0
                        .slice_move(s!(NewAxis))
                        .wrap()
                        .to_dimd()
                        .unwrap()
                        .into()
                } else {
                    arr
                }
            },
            RefType::Keep,
        )
    }

    /// Return a transposed view of the array.
    ///
    /// # Safety
    ///
    /// the data for the array view should exist
    pub unsafe fn t(self) -> Self {
        self.chain_view_f(
            move |arr| {
                let out: ArbArray<'_, T> = arr.0.t().wrap().into();
                mem::transmute(out)
                // arr.0.t().wrap().into()
            },
            RefType::True,
        )
    }

    /// Return a view of the diagonal elements of the array.
    ///
    /// The diagonal is simply the sequence indexed by
    /// (0, 0, .., 0), (1, 1, ..., 1) etc as long as all axes have elements.
    ///
    /// # Safety
    ///
    /// the data for the array view should exist
    pub unsafe fn diag(self) -> Self {
        self.chain_view_f(
            move |arr| {
                let out: ArbArray<'_, T> = arr.0.diag().wrap().to_dimd().unwrap().into();
                mem::transmute(out)
            },
            RefType::True,
        )
    }

    /// take values using slice
    ///
    /// # Safety
    ///
    /// the data for the array view should exist
    pub unsafe fn view_by_slice(self, mut slc: Vec<SliceInfoElem>) -> Self
    where
        T: Clone,
    {
        self.chain_view_f(
            move |arr| {
                // adjust the slice if start or end is greater than the length of the axis
                let mut axis = 0;
                let shape = arr.shape();
                slc.iter_mut().for_each(|elem| {
                    match elem {
                        SliceInfoElem::Slice { start, end, step } => {
                            let len = shape[axis] as isize;
                            // adjust when step is smaller than zero so the result when step < 0 is the same with numpy slice
                            if *step < 0 {
                                if *start < 0 {
                                    *start += len;
                                }
                                if end.is_some() {
                                    let mut _end = *end.as_ref().unwrap();
                                    if _end < 0 {
                                        _end += len;
                                    } else if _end >= len {
                                        _end = len;
                                    }
                                    if *start > _end {
                                        if *start == len {
                                            *end = Some(len);
                                        } else {
                                            *end = Some(*start + 1)
                                        };
                                        if _end >= len - 2 {
                                            *start = len - 1;
                                        } else {
                                            *start = _end + 1;
                                        }
                                    } else {
                                        // if start < end and step < 0, we shouldn't slice anything
                                        (*start, *end) = (0, Some(0));
                                    }
                                }
                            } else if let Some(_end) = end {
                                if *_end >= len {
                                    *end = Some(len)
                                }
                            }
                            axis += 1;
                        }
                        SliceInfoElem::Index(_) => axis += 1,
                        _ => {}
                    }
                });
                let slc_len = slc.len();
                if slc_len < arr.ndim() {
                    for _idx in slc_len..arr.ndim() {
                        slc.push(SliceInfoElem::Slice {
                            start: 0,
                            end: None,
                            step: 1,
                        })
                    }
                }
                let slc_info = unsafe {
                    SliceInfo::new_unchecked(
                        slc.as_slice(),
                        PhantomData::<IxDyn>,
                        PhantomData::<IxDyn>,
                    )
                };
                let out: ArbArray<'_, T> = arr.0.slice_move(slc_info).wrap().into();
                mem::transmute(out)
                // arr.0.slice_move(slc_info).wrap().into()
            },
            RefType::True,
        )
    }

    /// Swap axes ax and bx.
    ///
    /// This does not move any data, it just adjusts the array’s dimensions and strides.
    ///
    /// # Safety
    ///
    /// the data for the array view should exist
    pub unsafe fn swap_axes(self, ax: i32, bx: i32) -> Self {
        self.chain_view_f(
            move |arr| {
                let ax = arr.norm_axis(ax);
                let bx = arr.norm_axis(bx);
                let mut arr = arr.0;
                arr.swap_axes(ax.index(), bx.index());
                let out: ArbArray<'_, T> = arr.wrap().into();
                mem::transmute(out)
            },
            RefType::True,
        )
    }

    /// Permute the axes.
    ///
    /// This does not move any data, it just adjusts the array’s dimensions and strides.
    ///
    /// i in the j-th place in the axes sequence means self's i-th axis becomes self.permuted_axes()'s j-th axis
    ///
    /// # Safety
    ///
    /// the data for the array view should exist
    pub unsafe fn permuted_axes(self, axes: Expr<'a, usize>) -> Self {
        self.chain_view_f(
            move |arr| {
                let axes = axes.eval();
                let axes_view = axes.view_arr().to_dim1().expect("axes should be dim 1");
                let out: ArbArray<'_, T> = arr
                    .0
                    .permuted_axes(axes_view.to_slice().unwrap())
                    .wrap()
                    .into();
                mem::transmute(out)
                // arr.0.permuted_axes(axes_view.to_slice().unwrap()).wrap().into()
            },
            RefType::True,
        )
    }

    /// Insert new array axis at axis and return the result.
    ///
    /// # Safety
    ///
    /// the data for the array view should exist
    pub unsafe fn insert_axis(self, axis: i32) -> Self {
        self.chain_view_f(
            move |arr| {
                let axis = arr.norm_axis(axis);
                let out: ArbArray<'_, T> = arr.0.insert_axis(axis).wrap().into();
                mem::transmute(out)
            },
            RefType::True,
        )
    }

    /// Remove new array axis at axis and return the result.
    ///
    /// # Safety
    ///
    /// the data for the array view should exist
    pub unsafe fn remove_axis(self, axis: i32) -> Self {
        self.chain_view_f(
            move |arr| {
                let axis = arr.norm_axis(axis);
                let out: ArbArray<'_, T> = arr.0.remove_axis(axis).wrap().into();
                mem::transmute(out)
            },
            RefType::True,
        )
    }

    /// # Safety
    ///
    /// the data for the array view should exist
    pub unsafe fn broadcast(self, shape: Expr<'a, usize>) -> Self {
        self.chain_view_f(
            move |arr| {
                let shape = shape.eval();
                let sh = shape.view_arr();
                let ndim = sh.ndim();
                let out: ArbArray<'_, T> = if ndim == 0 {
                    let shape = sh.to_dim0().unwrap().into_scalar();
                    arr.broadcast(*shape)
                        .expect("broadcast error")
                        .to_dimd()
                        .unwrap()
                        .into()
                } else if ndim == 1 {
                    let shape = sh.to_dim1().unwrap();
                    arr.broadcast(shape.to_slice().unwrap())
                        .expect("broadcast error")
                        .to_dimd()
                        .unwrap()
                        .into()
                } else {
                    panic!("the dim of shape should not be greater than 1")
                };
                mem::transmute(out)
            },
            RefType::True,
        )
    }

    /// # Safety
    ///
    /// the data for the array view should exist
    pub unsafe fn broadcast_with<T2: ExprElement + 'a>(self, other: Expr<'a, T2>) -> Self {
        let shape = other.shape();
        self.broadcast(shape)
    }

    pub fn if_then(self, con: Expr<'a, bool>, then: Expr<'a, T>) -> Expr<'a, T>
    where
        T: Clone,
    {
        self.chain_arr_f(
            move |arr| {
                let con = con.eval();
                let flag = con
                    .view_arr()
                    .to_dim0()
                    .expect("condition should be a bool");
                if *flag.into_scalar() {
                    then.eval().into_arr().to_owned().into()
                } else {
                    arr
                }
            },
            RefType::Keep,
        )
    }
}
