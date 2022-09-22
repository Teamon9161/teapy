use super::prelude::*;
// use std::slice::GroupBy;

impl<T, S, D> ArrBase<S, D>
where
    S: Data<Elem = T>,
    D: Dimension,
{
    // pub fn groupby<S2, F>(&self, by: ArrayBase<S2, D>, eq_f: F) -> GroupBy<'_, T, F>
    // where
    //     F: FnMut(&T, &T) -> bool,
    //     S2: Data<Elem = T>,
    // {
    //     let a = vec![1,2,3];
    //     if let Some(slc) = self.as_slice_memory_order() {
    //         return slc.group_by(|a, b| a == b)
    //     } else {
    //         let arr_copy = self.0.to_owned();
    //     }
    // }
}
