/// define the compensation to use in kahan summation.
macro_rules! define_c {
    ($($c: ident),*) => {
        $(let $c = &mut 0.;)*
    }
}
pub(crate) use define_c;
