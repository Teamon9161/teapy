use super::{Expr, ExprElement, RefType};
use std::cmp::{PartialEq, PartialOrd};

// Impl expressions
impl<'a, T> Expr<'a, T>
where
    T: ExprElement + 'a,
{
    pub fn gt<T2>(self, rhs: Expr<'a, T2>, par: bool) -> Expr<'a, bool>
    where
        T: PartialOrd<T2>,
        T2: ExprElement + 'a,
    {
        self.chain_view_f(
            move |arr| Ok(arr.gt(rhs.eval()?.view().try_as_arr()?, par).into()),
            RefType::False,
        )
    }

    pub fn ge<T2>(self, rhs: Expr<'a, T2>, par: bool) -> Expr<'a, bool>
    where
        T: PartialOrd<T2>,
        T2: ExprElement + 'a,
    {
        self.chain_view_f(
            move |arr| Ok(arr.ge(rhs.eval()?.view().try_as_arr()?, par).into()),
            RefType::False,
        )
    }

    pub fn lt<T2>(self, rhs: Expr<'a, T2>, par: bool) -> Expr<'a, bool>
    where
        T: PartialOrd<T2>,
        T2: ExprElement + 'a,
    {
        self.chain_view_f(
            move |arr| Ok(arr.lt(rhs.eval()?.view().try_as_arr()?, par).into()),
            RefType::False,
        )
    }

    pub fn le<T2>(self, rhs: Expr<'a, T2>, par: bool) -> Expr<'a, bool>
    where
        T: PartialOrd<T2>,
        T2: ExprElement + 'a,
    {
        self.chain_view_f(
            move |arr| Ok(arr.le(rhs.eval()?.view().try_as_arr()?, par).into()),
            RefType::False,
        )
    }

    pub fn eq<T2>(self, rhs: Expr<'a, T2>, par: bool) -> Expr<'a, bool>
    where
        T: PartialEq<T2>,
        T2: ExprElement + 'a,
    {
        self.chain_view_f(
            move |arr| Ok(arr.eq(rhs.eval()?.view().try_as_arr()?, par).into()),
            RefType::False,
        )
    }

    pub fn ne<T2>(self, rhs: Expr<'a, T2>, par: bool) -> Expr<'a, bool>
    where
        T: PartialEq<T2>,
        T2: ExprElement + 'a,
    {
        self.chain_view_f(
            move |arr| Ok(arr.ne(rhs.eval()?.view().try_as_arr()?, par).into()),
            RefType::False,
        )
    }
}
