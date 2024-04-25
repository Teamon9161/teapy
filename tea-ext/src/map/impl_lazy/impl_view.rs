#![allow(clippy::missing_transmute_annotations)]

use lazy::{adjust_slice, Expr};
use ndarray::SliceInfoElem;
use std::mem::transmute;
use tea_core::prelude::*;

#[ext_trait]
impl<'a> ExprViewExt for Expr<'a> {
    #[allow(unreachable_patterns, clippy::missing_transmute_annotations)]
    fn index_axis(&mut self, index: Expr<'a>, axis: i32) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            let axis = arr.norm_axis(axis);
            let index = index
                .view_arr(ctx.as_ref())?
                .deref()
                .cast_i32()
                .into_owned()
                .into_scalar()?;
            match_arrok!(arr, arr, {
                let index = arr.view().norm_index(index, axis);
                let view = unsafe { transmute(arr.view().index_axis(axis, index).wrap()) };
                let out: ArrOk<'a> = ViewOnBase::new(arr, view).into();
                Ok((out.into(), ctx))
            })
        });
        self
    }

    #[allow(unreachable_patterns)]
    fn reshape(&mut self, shape: Expr<'a>) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            let shape = shape.view_arr(ctx.as_ref())?.deref().cast_usize();
            let out: ArrOk<'a> = match_arrok!(arr, arr, {
                let ndim = shape.ndim();
                if ndim == 0 {
                    let shape = shape.into_owned().to_dim0()?.into_scalar();
                    let view = unsafe {
                        transmute(
                            arr.view()
                                .0
                                .into_shape(shape)
                                .map_err(|e| StrError::from(format!("{e:?}")))?
                                .wrap()
                                .to_dimd(),
                        )
                    };
                    ViewOnBase::new(arr, view).into()
                } else if ndim == 1 {
                    let shape = shape.view().to_dim1()?;
                    let view = unsafe {
                        transmute(
                            arr.view()
                                .0
                                .into_shape(shape.to_slice().unwrap())
                                .map_err(|e| StrError::from(format!("{e:?}")))?
                                .wrap()
                                .to_dimd(),
                        )
                    };

                    ViewOnBase::new(arr, view).into()
                } else {
                    return Err("shape must be 0 or 1 dim".into());
                }
            });
            Ok((out.into(), ctx))
        });
        self
    }

    /// Return a transposed view of the array.
    #[allow(unreachable_patterns)]
    fn t(&mut self) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            match_arrok!(arr, arr, {
                let view = unsafe { transmute(arr.view().0.t().wrap()) };
                let out: ArrOk<'a> = ViewOnBase::new(arr, view).into();
                Ok((out.into(), ctx))
            })
        });
        self
    }

    /// Return a view of the diagonal elements of the array.
    ///
    /// The diagonal is simply the sequence indexed by
    /// (0, 0, .., 0), (1, 1, ..., 1) etc as long as all axes have elements.
    #[allow(unreachable_patterns)]
    fn diag(&mut self) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            match_arrok!(arr, arr, {
                let view = unsafe { transmute(arr.view().0.diag().wrap().to_dimd()) };
                let out: ArrOk<'a> = ViewOnBase::new(arr, view).into();
                Ok((out.into(), ctx))
            })
        });
        self
    }

    /// Swap axes ax and bx.
    ///
    /// This does not move any data, it just adjusts the array’s dimensions and strides.
    #[allow(unreachable_patterns)]
    fn swap_axes(&mut self, ax: i32, bx: i32) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            match_arrok!(arr, arr, {
                let mut view: ArrViewD<_> = unsafe { transmute(arr.view()) };
                let ax = view.norm_axis(ax).index();
                let bx = view.norm_axis(bx).index();
                view.0.swap_axes(ax, bx);
                let out: ArrOk<'a> = ViewOnBase::new(arr, view).into();
                Ok((out.into(), ctx))
            })
        });
        self
    }

    /// Permute the axes.
    ///
    /// This does not move any data, it just adjusts the array’s dimensions and strides.
    ///
    /// i in the j-th place in the axes sequence means self's i-th axis becomes self.permuted_axes()'s j-th axis
    #[allow(unreachable_patterns)]
    fn permuted_axes(&mut self, axes: Expr<'a>) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            let axes = axes.view_arr(ctx.as_ref())?.deref().cast_i32();
            match_arrok!(arr, arr, {
                let axes = axes
                    .view()
                    .to_dim1()?
                    .map(|axis| arr.view().norm_axis(*axis).0);
                let view = unsafe {
                    transmute(
                        arr.view()
                            .0
                            .permuted_axes(axes.view().to_slice().unwrap())
                            .wrap(),
                    )
                };
                let out: ArrOk<'a> = ViewOnBase::new(arr, view).into();
                Ok((out.into(), ctx))
            })
        });
        self
    }

    /// Insert new array axis at axis and return the result.

    #[allow(unreachable_patterns)]
    fn insert_axis(&mut self, axis: i32) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            match_arrok!(arr, arr, {
                let axis = arr.view().norm_axis(axis);
                let view = unsafe { transmute(arr.view().0.insert_axis(axis).wrap()) };
                let out: ArrOk<'a> = ViewOnBase::new(arr, view).into();
                Ok((out.into(), ctx))
            })
        });
        self
    }

    /// Remove new array axis at axis and return the result.

    #[allow(unreachable_patterns)]
    fn remove_axis(&mut self, axis: i32) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            match_arrok!(arr, arr, {
                let axis = arr.view().norm_axis(axis);
                let view = unsafe { transmute(arr.view().0.remove_axis(axis).wrap()) };
                let out: ArrOk<'a> = ViewOnBase::new(arr, view).into();
                Ok((out.into(), ctx))
            })
        });
        self
    }

    /// broadcast to a given shape

    #[allow(unreachable_patterns)]
    fn broadcast(&mut self, shape: Expr<'a>) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            let shape = shape.view_arr(ctx.as_ref())?.deref().cast_usize();
            let out: ArrOk<'a> = match_arrok!(arr, arr, {
                let shape_view = shape.view();
                let ndim = shape.ndim();
                if ndim == 0 {
                    let shape = shape_view.to_dim0()?.into_scalar();
                    let view = unsafe {
                        transmute(
                            arr.view()
                                .0
                                .broadcast(*shape)
                                .ok_or(StrError::from(format!(
                                    "Can not broadcast to shape: {shape:?}"
                                )))?
                                .wrap()
                                .to_dimd(),
                        )
                    };
                    ViewOnBase::new(arr, view).into()
                } else if ndim == 1 {
                    let shape = shape_view.to_dim1()?;
                    let view = unsafe {
                        transmute(
                            arr.view()
                                .0
                                .broadcast(shape.to_slice().unwrap())
                                .ok_or(StrError::from(format!(
                                    "Can not broadcast to shape: {shape:?}"
                                )))?
                                .wrap()
                                .to_dimd(),
                        )
                    };

                    ViewOnBase::new(arr, view).into()
                } else {
                    return Err("shape must be 0 or 1 dim".into());
                }
            });
            Ok((out.into(), ctx))
        });
        self
    }

    #[cfg(feature = "agg")]
    fn broadcast_with(&mut self, mut other: Expr<'a>) -> &mut Self {
        use crate::ExprAggExt;
        let shape = other.shape();
        self.broadcast(shape.clone());
        self
    }

    fn if_then(&mut self, con: Expr<'a>, then: Expr<'a>) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            let flag = con
                .view_arr(ctx.as_ref())?
                .deref()
                .cast_bool()
                .into_owned()
                .into_scalar()?;
            let then = then.clone().into_arr(ctx.clone())?;
            if flag {
                Ok((then.into(), ctx))
            } else {
                Ok((arr.into(), ctx))
            }
        });
        self
    }

    /// take values using slice
    #[allow(unreachable_patterns)]
    fn view_by_slice(&mut self, slc: Vec<SliceInfoElem>) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            let slc_info = adjust_slice(slc.clone(), arr.shape(), arr.ndim());
            match_arrok!(arr, arr, {
                let view = unsafe { transmute(arr.view().0.slice_move(slc_info).wrap()) };
                let out: ArrOk<'a> = ViewOnBase::new(arr, view).into();
                Ok((out.into(), ctx))
            })
        });
        self
    }
}
