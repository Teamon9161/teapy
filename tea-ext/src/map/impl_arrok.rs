use super::*;
use std::cmp::Ordering;
use tea_core::prelude::*;

#[ext_trait]
impl<'a> ArrOkExt for ArrOk<'a> {
    #[allow(unreachable_patterns, clippy::collapsible_else_if)]
    fn select(&self, slc: &Self, axis: i32, check: bool) -> TpResult<ArrOk<'a>> {
        if slc.ndim() > 1 {
            return Err("The slice must be dim 0 or dim 1 when select on axis".into());
        }
        let mut slc = slc.deref();
        let out = match_arrok!(self, a, {
            let a_view = a.view();
            let axis_ = a_view.norm_axis(axis);
            let length = a_view.len_of(axis_);
            if matches!(&slc, ArrOk::OptUsize(_)) {
                // take option_usize
                let slc = slc.cast_optusize();
                let slc_view = slc.view();
                match a_view.dtype() {
                    DataType::I32 => {
                        if slc_view.len() == 1 {
                            unsafe { a_view.into_dtype::<i32>() }
                                .cast::<f64>()
                                .index_axis(axis_, slc_view.to_dim1()?[0].unwrap())
                                .to_owned()
                                .wrap()
                                .into()
                        } else {
                            unsafe {
                                a_view
                                    .into_dtype::<i32>()
                                    .cast::<f64>()
                                    .take_option_clone_unchecked(
                                        slc_view.to_dim1()?,
                                        axis_.index() as i32,
                                        false,
                                    )
                            }
                            .into()
                        }
                    }
                    DataType::I64 => {
                        if slc_view.len() == 1 {
                            unsafe { a_view.into_dtype::<i64>() }
                                .cast::<f64>()
                                .index_axis(axis_, slc_view.to_dim1()?[0].unwrap())
                                .to_owned()
                                .wrap()
                                .into()
                        } else {
                            unsafe {
                                a_view
                                    .into_dtype::<i64>()
                                    .cast::<f64>()
                                    .take_option_clone_unchecked(
                                        slc_view.to_dim1()?,
                                        axis_.index() as i32,
                                        false,
                                    )
                            }
                            .into()
                        }
                    }
                    _ => {
                        if slc_view.len() == 1 {
                            a_view
                                .index_axis(axis_, slc_view.to_dim1()?[0].unwrap())
                                .to_owned()
                                .wrap()
                                .into()
                        } else {
                            unsafe {
                                a_view.take_option_clone_unchecked(
                                    slc_view.to_dim1()?,
                                    axis_.index() as i32,
                                    false,
                                )
                            }
                            .into()
                        }
                    }
                }
            } else if matches!(&slc, ArrOk::Bool(_)) {
                let slc = slc.cast_bool();
                let slc_view = slc.view();
                a_view.filter(&slc_view.to_dim1()?, axis, false).into()
            } else {
                if check {
                    slc = match slc {
                        ArrOk::I32(slc) => slc
                            .deref()
                            .view()
                            .to_dim1()?
                            .map(|s| a_view.ensure_index(*s, length))
                            .to_dimd()
                            .into(),
                        ArrOk::I64(slc) => slc
                            .deref()
                            .view()
                            .to_dim1()?
                            .map(|s| a_view.ensure_index(*s as i32, length))
                            .to_dimd()
                            .into(),
                        _ => slc,
                    };
                }
                let slc = slc.cast_usize();
                let slc_view = slc.view();
                if slc_view.len() == 1 {
                    a_view
                        .index_axis(axis_, slc_view.to_dim1()?[0])
                        .to_owned()
                        .wrap()
                        .into()
                } else {
                    if check {
                        a_view
                            .select(axis_, slc_view.to_dim1()?.as_slice().unwrap())
                            .wrap()
                            .into()
                    } else {
                        a_view
                            .select_unchecked(axis_, slc_view.to_dim1()?.as_slice().unwrap())
                            .into()
                    }
                }
            }
        });
        Ok(out)
    }

    fn get_sort_idx<'r>(by: &'r [&'r ArrOk<'a>], rev: bool) -> TpResult<Vec<usize>> {
        // if self.ndim() != 1 {
        //     return Err("Currently only 1 dim Expr can be sorted".into());
        // }
        let len = by[0].len();
        let mut idx = Vec::from_iter(0..len);
        use ArrOk::*;
        idx.sort_by(move |a, b| {
            let mut order = Ordering::Equal;
            for arr in by.iter() {
                let rtn = match &arr {
                    String(_) => {
                        match_arrok!(
                            arr,
                            arr,
                            {
                                let key_view = arr
                                    .view()
                                    .to_dim1()
                                    .expect("Currently only 1 dim array can be sort key");
                                let (va, vb) = unsafe { (key_view.uget(*a), key_view.uget(*b)) };
                                if !rev {
                                    va.cmp(vb)
                                } else {
                                    va.cmp(vb).reverse()
                                }
                            },
                            String // DateTime
                        )
                    }
                    #[cfg(feature = "time")]
                    DateTime(_) => {
                        match_arrok!(
                            arr,
                            arr,
                            {
                                let key_view = arr
                                    .view()
                                    .to_dim1()
                                    .expect("Currently only 1 dim array can be sort key");
                                let (va, vb) = unsafe { (key_view.uget(*a), key_view.uget(*b)) };
                                if !rev {
                                    va.cmp(vb)
                                } else {
                                    va.cmp(vb).reverse()
                                }
                            },
                            DateTime // TimeDelta
                        )
                    }
                    _ => {
                        match_arrok!(
                            numeric arr,
                            arr,
                            {
                                let key_view = arr.view().to_dim1().expect(
                                    "Currently only 1 dim array can be sort key",
                                );
                                let (va, vb) =
                                    unsafe { (key_view.uget(*a), key_view.uget(*b)) };
                                if !rev {
                                    va.nan_sort_cmp_stable(vb)
                                } else {
                                    va.nan_sort_cmp_rev_stable(vb)
                                }
                            }
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
        Ok(idx)
    }
}
