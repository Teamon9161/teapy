/// define the compensation to use in kahan summation.
#[macro_export]
macro_rules! define_c {
    ($($c: ident),*) => {
        $(let $c = &mut 0.;)*
    }
}

// pub(super) use define_c;
#[macro_export]
macro_rules! define_n {
    ($($n: ident),*) => {
        $(let $n = &mut 0usize;)*
    }
}
// pub(super) use define_n;
