macro_rules! define_arrview {
    ($arr: ident, $from: ident) => {
        pub struct $arr<'a, T>(pub $from<'a, T>);
        impl<'a, T> Deref for $arr<'a, T> {
            type Target = $from<'a, T>;
            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
    };
}

macro_rules! impl_arrview {
    ([$($arr: ident),*], $body: tt) => {
        $(impl<T> $arr<'_, T> $body)*
    };
    ([$($arr: ident),*], $bound: ident, $body: tt) => {
        $(impl<T: $bound> $arr<'_, T> $body)*
    };
    ([$($arr: ident),*], $bound: ident + $bound1: ident, $body: tt) => {
        $(impl<T: $bound + $bound1 > $arr<'_, T> $body)*
    }
}
