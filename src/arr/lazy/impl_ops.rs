use super::{Expr, ExprElement, RefType};
use crate::arr::{DateTime, TimeDelta, WrapNdarray};
use ndarray::{ScalarOperand, Zip};
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, Mul, MulAssign, Neg,
    Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

macro_rules! impl_binary_op {
    ($trt: ident, $operator: tt, $func: ident, $assign_trt: ident, $assign_func: ident) => {
        impl<'a, T, T2> $trt<Expr<'a, T2>> for Expr<'a, T>
        where
            T: ExprElement + $trt<T2, Output=T> + Clone + ScalarOperand,
            T2: ExprElement + Clone + ScalarOperand,
        {
            type Output = Expr<'a, T>;
            fn $func(self, other: Expr<'a, T2>) -> Self {
                self.chain_owned_f(move |arr| (arr.0 $operator other.eval().view_arr().0).wrap().into())
            }
        }

        impl<'a, T, T2> $assign_trt<Expr<'a, T2>> for Expr<'a, T>
        where
            T: ExprElement + $trt<T2, Output=T> + Clone + ScalarOperand,
            T2: ExprElement + Clone + ScalarOperand,
        {
            fn $assign_func(&mut self, other: Expr<'a, T2>) {
                *self = std::mem::take(self) $operator other;
            }
        }
    };
}

impl_binary_op!(Add, +, add, AddAssign, add_assign);
impl_binary_op!(Sub, -, sub, SubAssign, sub_assign);
impl_binary_op!(Mul, *, mul, MulAssign, mul_assign);
impl_binary_op!(Div, /, div, DivAssign, div_assign);
impl_binary_op!(Rem, %, rem, RemAssign, rem_assign);
impl_binary_op!(BitAnd, &, bitand, BitAndAssign, bitand_assign);
impl_binary_op!(BitOr, |, bitor, BitOrAssign, bitor_assign);
impl_binary_op!(Shl, <<, shl, ShlAssign, shl_assign);
impl_binary_op!(Shr, >>, shr, ShrAssign, shr_assign);

impl<'a, T> Neg for Expr<'a, T>
where
    T: ExprElement + Neg<Output = T> + Clone + 'a,
{
    type Output = Self;
    fn neg(self) -> Self {
        self.chain_owned_f(move |arr| (-arr.0).wrap().into())
    }
}

impl<'a, T> Not for Expr<'a, T>
where
    T: ExprElement + Not<Output = T> + Clone + 'a,
{
    type Output = Self;
    fn not(self) -> Self {
        self.chain_owned_f(move |arr| (!arr.0).wrap().into())
    }
}

impl<'a> Expr<'a, DateTime> {
    pub fn sub_datetime(self, other: Expr<'a, DateTime>, par: bool) -> Expr<'a, TimeDelta> {
        self.chain_view_f(
            move |arr| {
                if !par {
                    Zip::from(arr.0)
                        .and(other.eval().view_arr().0)
                        .map_collect(|v1, v2| *v1 - *v2)
                        .wrap()
                        .into()
                } else {
                    Zip::from(arr.0)
                        .and(other.eval().view_arr().0)
                        .par_map_collect(|v1, v2| *v1 - *v2)
                        .wrap()
                        .into()
                }
            },
            RefType::False,
        )
    }
}
