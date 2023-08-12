use super::{Expr, ExprElement};

impl<'a, T> Expr<'a, T>
where
    T: ExprElement + Clone + 'a,
{
    pub fn put_mask(self, mask: Expr<'a, bool>, value: Expr<'a, T>, axis: i32, par: bool) -> Self {
        self.chain_view_mut_f(move |arr| {
            arr.put_mask(&mask.eval().view_arr(), &value.eval().view_arr(), axis, par);
        })
    }
}
