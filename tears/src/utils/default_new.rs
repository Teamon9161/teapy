// pub trait DefaultNew {
//     fn default_new() -> Self;
// }

pub trait EmptyNew {
    fn empty_new() -> Self;
    // where
    //     Self: Sized,
    // {
    //     DefaultNew::default_new()
    // }
}
