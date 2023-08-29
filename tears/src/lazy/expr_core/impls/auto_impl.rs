use super::super::Expr;
use crate::{match_all, match_arrok, ArrOk};

macro_rules! auto_impl_view {
    (in1, $func:ident, ($($p:ident: $p_ty:ty),* $(,)?)) => {
        impl<'a> Expr<'a> {
            pub fn $func(&mut self $(, $p: $p_ty)*) {
                self.chain_f(move |data| {
                    let arr = data.into_arr()?;
                    match_arrok!(numeric arr, a, { Ok(a.view().$func($($p),*).into()) })
                })
            }
        }
    };
    (in1, [$($func: ident),* $(,)?], $other: tt) => {
        $(auto_impl_view!(in1, $func, $other);)*
    };
    // (in1-inplace, $func:ident, $func_inplace: ident -> $otype:ident, ($($p:ident: $p_ty:ty),* $(,)?)) => {
    //     pub fn $func (self $(, $p: $p_ty)*) -> Expr<'a, $otype> {
    //         self.chain_arr_f(move |arb_arr| {
    //             use ArbArray::*;
    //             match arb_arr {
    //                 View(arr) => Ok(arr.$func($($p),*).into()),
    //                 ViewMut(mut arr) => {
    //                     arr.$func_inplace($($p),*);
    //                     Ok(ViewMut(arr))
    //                 },
    //                 Owned(mut arr) => {
    //                     arr.view_mut().$func_inplace($($p),*);
    //                     Ok(Owned(arr))
    //                 },
    //             }
    //         }, RefType::Keep)
    //     }
    // };
    // (in1-inplace, [$($func: ident, $func_inplace: ident -> $otype:ident),* $(,)?], $other: tt) => {
    //     $(impl_view_lazy!(in1-inplace, $func, $func_inplace -> $otype, $other);)*
    // };

    // (in2, $func:ident -> $otype:ty, ($($p:ident: $p_ty:ty),* $(,)?)) => {
    //     pub fn $func<T2> (self, other: Expr<'a, T2> $(, $p: $p_ty)*) -> Expr<'a, $otype>
    //     where
    //         T2: Number + ExprElement,
    //     {
    //         self.chain_view_f_ct(move |(arr, ct)| {
    //             let (out, ct)= other.eval(ct)?;
    //             Ok((arr.$func(&out.view_arr(), $($p),*).into(), ct))
    //         }, RefType::False)
    //     }
    // };
    // (in2, [$($func: ident -> $otype:ident),* $(,)?], $other: tt) => {
    //     $(impl_view_lazy!(in2, $func -> $otype, $other);)*
    // };
}

// impl<'a> Expr<'a> {
//     pub fn is_nan(&mut self) {
//         self.chain_f(|data| {
//             let arr = data.into_arr()?;
//             match_arrok!(numeric arr, a, { Ok(a.view().is_nan().into()) })
//         })
//     }
// }

auto_impl_view!(in1, [is_nan, not_nan], ());
auto_impl_view!(in1, [diff, pct_change], (n: i32, axis: i32, par: bool));
auto_impl_view!(in1, [count_nan, count_notnan, median, max, min, prod, cumprod], (axis: i32, par: bool));
auto_impl_view!(in1, [sum, mean, var, std, skew, kurt, cumsum], (stable: bool, axis: i32, par: bool));
