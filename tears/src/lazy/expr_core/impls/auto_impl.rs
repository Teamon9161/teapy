use super::super::{Expr, FuncOut};
// use super::FuncOut;
use crate::{match_all, match_arrok, ArbArray, ArrOk, Cast, CorrMethod, WinsorizeMethod};

macro_rules! auto_impl_view {
    (in1, [$($func: ident),* $(,)?], $other: tt) => {
        $(auto_impl_view!(in1, $func, $other);)*
    };
    (in1, $func:ident, ($($p:ident: $p_ty:ty),* $(,)?)) => {
        impl<'a> Expr<'a> {
            pub fn $func(&mut self $(, $p: $p_ty)*) {
                self.chain_f_ctx(move |(mut data, ctx)| {
                    let (arr, ctx) = data.into_arr(ctx)?;
                    match_arrok!(numeric arr, a, { Ok((a.view().$func($($p),*).into(), ctx)) })
                })
            }
        }
    };
    (in2, [$($func: ident),* $(,)?], $other: tt) => {
        $(auto_impl_view!(in2, $func, $other);)*
    };
    (in2, $func:ident, ($($p:ident: $p_ty:ty),* $(,)?)) => {
        impl<'a> Expr<'a> {
            pub fn $func(&mut self, other: &mut Expr<'a>, $($p: $p_ty),*) {
                self.chain_f_ctx(move |(mut data, ctx): FuncOut<'a>| {
                    let ctx = other.eval_inplace(ctx)?;
                    let (arr, ctx) = data.into_arr(ctx)?;
                    let out = arr.cast_f64().view().$func(&other.base.view_arr().cast_f64().view(), $($p),*);
                    Ok((out.into(), ctx))
                    // match_arrok!(numeric arr, a, {
                    //     a.view().$func(&other_arr.cast(), $($p),*);
                    //     Ok((a.into(), ctx))
                    // })
                })
            }
        }
    };
}

macro_rules! auto_impl_viewmut {
    (in1, [$($func: ident),* $(,)?], $other: tt) => {
        $(auto_impl_viewmut!(in1, $func, $other);)*
    };
    (in1, $func:ident, ($($p:ident: $p_ty:ty),* $(,)?)) => {
        impl<'a> Expr<'a> {
            pub fn $func(&mut self, $($p: $p_ty),*) {
                self.chain_f_ctx(move |(mut data, ctx)| {
                    let (arr, ctx) = data.view_arr(ctx)?;
                    match_arrok!(numeric arr, a, {
                        a.viewmut().$func($($p),*);
                        Ok((a.viewmut().into(), ctx))
                    })
                })
            }
        }
    };
}

// impl<'a> Expr<'a> {
//     pub fn corr(&mut self, other: Expr<'a>, method: CorrMethod, stable: bool, axis: i32, par: bool) {
//         self.chain_f_ctx(move |(data, ctx): FuncOut<'a>| {
//             let (other_arr, ctx) = other.into_arr(ctx)?;
//             let (arr, ctx) = data.into_arr(ctx)?;
//             match_arrok!(numeric arr, a, {
//                 a.view().corr(&Cast::<ArbArray<_>>::cast(other_arr).view(), method, stable, axis, par);
//                 Ok((a.into(), ctx))
//             })
//         })
//     }
// }

auto_impl_view!(in1, [is_nan, not_nan], ());
auto_impl_view!(in1, [diff, pct_change], (n: i32, axis: i32, par: bool));
auto_impl_view!(in1, [count_nan, count_notnan, median, max, min, prod, cumprod], (axis: i32, par: bool));
auto_impl_view!(in1, [sum, mean, var, std, skew, kurt, cumsum], (stable: bool, axis: i32, par: bool));
auto_impl_viewmut!(in1, [zscore_inplace], (stable: bool, axis: i32, par: bool));
auto_impl_viewmut!(in1, [winsorize_inplace], (method: WinsorizeMethod, method_params: Option<f64>, stable: bool, axis: i32, par: bool));
auto_impl_view!(in2, [corr], (method: CorrMethod, stable: bool, axis: i32, par: bool));
// auto_impl_view!(in2, [cov], (stable: bool, axis: i32, par: bool));
