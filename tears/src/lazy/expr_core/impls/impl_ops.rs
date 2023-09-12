use super::export::*;
use std::ops::{Add, BitAnd, BitOr, Div, Mul, Neg, Not, Sub};

macro_rules! impl_cmp {
    ($func: ident $(,$dtype: ident)*) => {
        impl<'a> Expr<'a> {
            pub fn $func(&mut self, rhs: Expr<'a>, par: bool) {
                self.chain_f_ctx(move |(data, ctx)| {
                    let ldtype = data.view_arr(ctx.as_ref())?.dtype();
                    let rdtype = rhs.view_arr(ctx.as_ref())?.dtype();
                    let out = if ldtype.is_float() | rdtype.is_float() {
                        let arr = data.view_arr(ctx.as_ref())?.as_float();
                        match_arrok!(float arr, a, {
                            let rhs_arr: ArbArray<_> = rhs.view_arr(ctx.as_ref())?.deref().cast();
                            a.view().$func(&rhs_arr.view(), par)
                        })
                    } else if ldtype.is_int() | rdtype.is_int() {
                        let arr = data.view_arr(ctx.as_ref())?.as_int();
                        match_arrok!(int arr, a, {
                            let rhs_arr: ArbArray<_> = rhs.view_arr(ctx.as_ref())?.deref().cast();
                            a.view().$func(&rhs_arr.view(), par)
                        })
                    } else {
                        let arr = data.view_arr(ctx.as_ref())?;
                        let rhs_arr = rhs.view_arr(ctx.as_ref())?;
                        match (arr, rhs_arr) {
                            $(
                                (ArrOk::$dtype(a), ArrOk::$dtype(r)) => {
                                    a.view().$func(&r.view(), par)
                                },
                            )*
                            _ => panic!("can not cmp this type of expression")
                        }
                    };
                    // use ArrOk::*;
                    // let out = match (&arr, &rhs_arr) {
                    //     (F64(_), _) | (_, F64(_)) => {
                    //         match_arrok!((arr, a, I32, I64, F32, F64, Usize), (rhs_arr, b, I32, I64, F32, F64, Usize), {
                    //             (a.cast_f64().view().$func(&b.cast_f64().view(), par))
                    //         })
                    //     },
                    //     (F32(_), _) | (_, F32(_)) => {
                    //         match_arrok!((arr, a, I32, I64, F32, Usize), (rhs_arr, b, I32, I64, F32, Usize), {
                    //             (a.cast_f32().view().$func(&b.cast_f32().view(), par))
                    //         })
                    //     },
                    //     (I64(_), _) | (_, I64(_)) => {
                    //         match_arrok!((arr, a, I64, I32, Bool, Usize), (rhs_arr, b, I64, I32, Bool, Usize), {
                    //             (a.cast_i64().view().$func(&b.cast_i64().view(), par))
                    //         })
                    //     },
                    //     (I32(_), _) | (_, I32(_)) => {
                    //         match_arrok!((arr, a, I32, Bool, Usize), (rhs_arr, b, I32, Bool, Usize), {
                    //             (a.cast_i32().view().$func(&b.cast_i32().view(), par))
                    //         })
                    //     },
                    //     // (Object(e1), String(e2)) => e1.clone().object_to_string($py).map_err(StrError::to_py)?.$func(e2.clone(), false).to_py(obj).add_obj_into(obj2),
                    //     // (String(e1), Object(e2)) => e1.clone().$func(e2.clone().object_to_string($py).map_err(StrError::to_py)?, false).to_py(obj).add_obj_into(obj2),
                        // $(
                        //     ($dtype(a), $dtype(r)) => a.view().$func(&r.view(), par),
                        // )*
                    //     _ => todo!()
                    // };
                    Ok((out.into(), ctx))
                });
            }
        }
    };
}

macro_rules! impl_dot {
    ($func: ident $(,$dtype: ident)*) => {
        impl<'a> Expr<'a> {
            pub fn $func(&mut self, rhs: Expr<'a>) {
                self.chain_f_ctx(move |(data, ctx)| {
                    let ldtype = data.view_arr(ctx.as_ref())?.dtype();
                    let rdtype = rhs.view_arr(ctx.as_ref())?.dtype();
                    let out: ArrOk = if ldtype.is_float() | rdtype.is_float() {
                        let arr = data.view_arr(ctx.as_ref())?.as_float();
                        match_arrok!(float arr, a, {
                            let rhs_arr: ArbArray<_> = rhs.view_arr(ctx.as_ref())?.deref().cast();
                            a.view().$func(&rhs_arr.view())?.into()
                        })
                    } else if ldtype.is_int() | rdtype.is_int() {
                        let arr = data.view_arr(ctx.as_ref())?.as_int();
                        match_arrok!(int arr, a, {
                            let rhs_arr: ArbArray<_> = rhs.view_arr(ctx.as_ref())?.deref().cast();
                            a.view().$func(&rhs_arr.view())?.into()
                        })
                    } else {
                        let arr = data.view_arr(ctx.as_ref())?;
                        let rhs_arr = rhs.view_arr(ctx.as_ref())?;
                        match (arr, rhs_arr) {
                            $(
                                (ArrOk::$dtype(a), ArrOk::$dtype(r)) => {
                                    a.view().$func(&r.view())?.into()
                                },
                            )*
                            _ => panic!("can not dot this type of expression")
                        }
                    };
                    // let arr = data.into_arr(ctx.clone())?;
                    // rhs.eval_inplace(ctx.clone())?;
                    // let rhs_arr = rhs.into_arr(None)?;
                    // use ArrOk::*;
                    // let out = match (&arr, &rhs_arr) {
                    //     (F64(_), _) | (_, F64(_)) => {
                    //         match_arrok!((arr, a, I32, I64, F32, F64, Usize), (rhs_arr, b, I32, I64, F32, F64, Usize), {
                    //             (a.cast_f64().view().$func(&b.cast_f64().view())?).into()
                    //         })
                    //     },
                    //     (F32(_), _) | (_, F32(_)) => {
                    //         match_arrok!((arr, a, I32, I64, F32, Usize), (rhs_arr, b, I32, I64, F32, Usize), {
                    //             (a.cast_f32().view().$func(&b.cast_f32().view())?).into()
                    //         })
                    //     },
                    //     (I64(_), _) | (_, I64(_)) => {
                    //         match_arrok!((arr, a, I64, I32, Bool, Usize), (rhs_arr, b, I64, I32, Bool, Usize), {
                    //             (a.cast_i64().view().$func(&b.cast_i64().view())?).into()
                    //         })
                    //     },
                    //     (I32(_), _) | (_, I32(_)) => {
                    //         match_arrok!((arr, a, I32, Bool, Usize), (rhs_arr, b, I32, Bool, Usize), {
                    //             (a.cast_i32().view().$func(&b.cast_i32().view())?).into()
                    //         })
                    //     },
                    //     $(
                    //         ($dtype(a), $dtype(r)) => a.view().$func(&r.view())?.into(),
                    //     )*
                    //     _ => todo!()
                    // };
                    Ok((out.into(), ctx))
                })
            }
        }
    };
}

impl_cmp!(eq, DateTime, String, TimeDelta, Bool);
impl_cmp!(ne, DateTime, String, TimeDelta, Bool);
impl_cmp!(gt, DateTime, String, TimeDelta);
impl_cmp!(ge, DateTime, String, TimeDelta);
impl_cmp!(lt, DateTime, String, TimeDelta);
impl_cmp!(le, DateTime, String, TimeDelta);

impl_dot!(dot);

impl<'a> Add for Expr<'a> {
    type Output = Expr<'a>;
    fn add(mut self, rhs: Self) -> Self::Output {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            let rhs_arr = rhs.view_arr(ctx.as_ref())?;
            use ArrOk::*;
            let out = match (&arr, &rhs_arr) {
                (F64(_), _) | (_, F64(_)) => (arr.cast_f64().into_owned().0
                    + rhs_arr.deref().cast_f64().view().0)
                    .wrap()
                    .into(),
                (F32(_), _) | (_, F32(_)) => (arr.cast_f32().into_owned().0
                    + rhs_arr.deref().cast_f32().view().0)
                    .wrap()
                    .into(),
                (I64(_), _) | (_, I64(_)) => (arr.cast_i64().into_owned().0
                    + rhs_arr.deref().cast_i64().view().0)
                    .wrap()
                    .into(),
                (I32(_), _) | (_, I32(_)) => (arr.cast_i32().into_owned().0
                    + rhs_arr.deref().cast_i32().view().0)
                    .wrap()
                    .into(),
                (Usize(_), _) | (_, Usize(_)) => (arr.cast_usize().into_owned().0
                    + rhs_arr.deref().cast_usize().view().0)
                    .wrap()
                    .into(),
                (String(_), String(_)) => arr
                    .cast_string()
                    .view()
                    .add_string(&rhs_arr.deref().cast_string().view())
                    .into(),
                (Str(_), String(_)) => arr
                    .cast_string()
                    .view()
                    .add_string(&rhs_arr.deref().cast_string().view())
                    .into(),
                (String(_), Str(_)) => arr
                    .cast_string()
                    .view()
                    .add_str(&rhs_arr.deref().cast_str().view())
                    .into(),
                (Str(_), Str(_)) => arr
                    .cast_string()
                    .view()
                    .add_str(&rhs_arr.deref().cast_str().view())
                    .into(),
                (DateTime(_), TimeDelta(_)) => arr
                    .cast_datetime_default()
                    .into_owned()
                    .0
                    .add(rhs_arr.deref().cast_timedelta().view().0)
                    .wrap()
                    .into(),
                (TimeDelta(_), DateTime(_)) => rhs_arr
                    .deref()
                    .cast_datetime_default()
                    .into_owned()
                    .0
                    .add(arr.cast_timedelta().view().0)
                    .wrap()
                    .into(),
                (TimeDelta(_), TimeDelta(_)) => arr
                    .cast_timedelta()
                    .into_owned()
                    .0
                    .add(rhs_arr.deref().cast_timedelta().view().0)
                    .wrap()
                    .into(),
                _ => todo!(),
            };
            Ok((out, ctx))
        });
        self
    }
}

impl<'a> Sub for Expr<'a> {
    type Output = Expr<'a>;
    fn sub(mut self, rhs: Self) -> Self::Output {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            // rhs.eval_inplace(ctx.clone())?;
            let rhs_arr = rhs.view_arr(ctx.as_ref())?;
            use ArrOk::*;
            let out = match (&arr, &rhs_arr) {
                (F64(_), _) | (_, F64(_)) => (arr.cast_f64().into_owned().0
                    - rhs_arr.deref().cast_f64().view().0)
                    .wrap()
                    .into(),
                (F32(_), _) | (_, F32(_)) => (arr.cast_f32().into_owned().0
                    - rhs_arr.deref().cast_f32().view().0)
                    .wrap()
                    .into(),
                (I64(_), _) | (_, I64(_)) => (arr.cast_i64().into_owned().0
                    - rhs_arr.deref().cast_i64().view().0)
                    .wrap()
                    .into(),
                (I32(_), _) | (_, I32(_)) => (arr.cast_i32().into_owned().0
                    - rhs_arr.deref().cast_i32().view().0)
                    .wrap()
                    .into(),
                (Usize(_), _) | (_, Usize(_)) => (arr.cast_usize().into_owned().0
                    - rhs_arr.deref().cast_usize().view().0)
                    .wrap()
                    .into(),
                (DateTime(_), TimeDelta(_)) => arr
                    .cast_datetime_default()
                    .into_owned()
                    .0
                    .sub(rhs_arr.deref().cast_timedelta().view().0)
                    .wrap()
                    .into(),
                (DateTime(_), DateTime(_)) => arr
                    .cast_datetime_default()
                    .view()
                    .sub_datetime(&(rhs_arr.deref().cast_datetime_default().view()), false)
                    .into(),
                (TimeDelta(_), TimeDelta(_)) => arr
                    .cast_timedelta()
                    .into_owned()
                    .0
                    .add(rhs_arr.deref().cast_timedelta().view().0)
                    .wrap()
                    .into(),
                _ => todo!(),
            };
            Ok((out, ctx))
        });
        self
    }
}

impl<'a> Mul for Expr<'a> {
    type Output = Expr<'a>;
    fn mul(mut self, rhs: Self) -> Self::Output {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            // rhs.eval_inplace(ctx.clone())?;
            let rhs_arr = rhs.view_arr(ctx.as_ref())?;
            use ArrOk::*;
            let out = match (&arr, &rhs_arr) {
                (F64(_), _) | (_, F64(_)) => (arr.cast_f64().into_owned().0
                    * rhs_arr.deref().cast_f64().view().0)
                    .wrap()
                    .into(),
                (F32(_), _) | (_, F32(_)) => (arr.cast_f32().into_owned().0
                    * rhs_arr.deref().cast_f32().view().0)
                    .wrap()
                    .into(),
                (I64(_), _) | (_, I64(_)) => (arr.cast_i64().into_owned().0
                    * rhs_arr.deref().cast_i64().view().0)
                    .wrap()
                    .into(),
                (I32(_), _) | (_, I32(_)) => (arr.cast_i32().into_owned().0
                    * rhs_arr.deref().cast_i32().view().0)
                    .wrap()
                    .into(),
                (Usize(_), _) | (_, Usize(_)) => (arr.cast_usize().into_owned().0
                    * rhs_arr.deref().cast_usize().view().0)
                    .wrap()
                    .into(),
                // (TimeDelta(_), I32(_)) => {
                //     arr.cast_timedelta().to_owned().0.mul(rhs_arr.cast_i32().view().0).wrap().into()
                // },
                _ => todo!(),
            };
            Ok((out, ctx))
        });
        self
    }
}

impl<'a> Div for Expr<'a> {
    type Output = Expr<'a>;
    fn div(mut self, rhs: Self) -> Self::Output {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            // rhs.eval_inplace(ctx.clone())?;
            let rhs_arr = rhs.view_arr(ctx.as_ref())?;
            use ArrOk::*;
            let out = match (&arr, &rhs_arr) {
                (F64(_), _) | (_, F64(_)) => (arr.cast_f64().into_owned().0
                    / rhs_arr.deref().cast_f64().view().0)
                    .wrap()
                    .into(),
                (F32(_), _) | (_, F32(_)) => (arr.cast_f32().into_owned().0
                    / rhs_arr.deref().cast_f32().view().0)
                    .wrap()
                    .into(),
                (I64(_), _)
                | (_, I64(_))
                | (I32(_), _)
                | (_, I32(_))
                | (Usize(_), _)
                | (_, Usize(_)) => (arr.cast_f64().into_owned().0
                    / rhs_arr.deref().cast_f64().view().0)
                    .wrap()
                    .into(),
                _ => todo!(),
            };
            Ok((out, ctx))
        });
        self
    }
}

impl<'a> BitAnd for Expr<'a> {
    type Output = Expr<'a>;
    fn bitand(mut self, rhs: Self) -> Self::Output {
        self.cast_bool().chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            let mut rhs = rhs.clone();
            rhs.cast_bool().eval_inplace(ctx.clone())?;
            let rhs_arr = rhs.view_arr(ctx.as_ref())?;
            let out = match_arrok!((arr, a, Bool), (rhs_arr, b, Bool), {
                a.cast_bool()
                    .into_owned()
                    .0
                    .bitand(&b.view().0)
                    .wrap()
                    .into()
            });
            Ok((out, ctx))
        });
        self
    }
}

impl<'a> BitOr for Expr<'a> {
    type Output = Expr<'a>;
    fn bitor(mut self, rhs: Self) -> Self::Output {
        self.cast_bool().chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            let mut rhs = rhs.clone();
            rhs.cast_bool().eval_inplace(ctx.clone())?;
            let rhs_arr = rhs.view_arr(ctx.as_ref())?;
            let out = match_arrok!((arr, a, Bool), (rhs_arr, b, Bool), {
                a.cast_bool()
                    .into_owned()
                    .0
                    .bitor(&b.view().0)
                    .wrap()
                    .into()
            });
            Ok((out, ctx))
        });
        self
    }
}

impl<'a> Neg for Expr<'a> {
    type Output = Expr<'a>;
    fn neg(mut self) -> Self::Output {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            let out = match_arrok!(
                arr,
                a,
                { (-a.into_owned().0).wrap().into() },
                F32,
                F64,
                I32,
                I64
            );
            Ok((out, ctx))
        });
        self
    }
}

impl<'a> Not for Expr<'a> {
    type Output = Expr<'a>;
    fn not(mut self) -> Self::Output {
        self.cast_bool().chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            let out = match_arrok!(arr, a, { (!a.into_owned().0).wrap().into() }, Bool);
            Ok((out, ctx))
        });
        self
    }
}
