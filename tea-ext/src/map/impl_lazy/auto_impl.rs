use lazy::Expr;

macro_rules! auto_impl_map_view {
    (
        $(in1, [$($func: ident),* $(,)?], $other: tt);*
        $(;in2, [$($func2: ident),* $(,)?], $other2: tt)*
        $(;float, [$($func3: ident),* $(,)?], $other3: tt)*
        $(;)?
    ) => {
        #[ext_trait]
        impl<'a> AutoExprMapExt for Expr<'a> {
            $($(auto_impl_view!(in1, $func, $other);)*)*
            $($(auto_impl_view!(in2, $func2, $other2);)*)*
            $($(auto_impl_f64_func!($func3, $other3);)*)*
        }
    };
}

macro_rules! auto_impl_map_viewmut {
    (
        $(in1, [$($func: ident),* $(,)?], $other: tt);*
        // $(in2, [$($func2: ident),* $(,)?], $other2: tt);*
        $(;)?
    ) => {
        #[ext_trait]
        impl<'a> AutoExprInplaceExt for Expr<'a> {
            $($(auto_impl_viewmut!(in1, $func, $other);)*)*
            // $($(auto_impl_viewmut!(in2, $func2, $other2);)*)*
        }
    };
}

// todo! diff

auto_impl_map_view!(
    in1, [is_nan, not_nan], ();
    in1, [pct_change], (n: i32, axis: i32, par: bool);
    in1, [cumprod], (axis: i32, par: bool);
    in1, [cumsum], (stable: bool, axis: i32, par: bool);
    in1, [rank], (pct: bool, rev: bool, axis: i32, par: bool);
    in1, [argsort], (rev: bool, axis: i32, par: bool);
    in1, [split_group], (group: usize, rev: bool, axis: i32, par: bool);
    in1, [arg_partition, partition], (kth: usize, sort: bool, rev: bool, axis: i32, par: bool);
    float, [
        sqrt, cbrt, ln, ln_1p, log2, log10, exp, exp2, exp_m1,
        acos, asin, atan, sin, cos, tan, ceil, floor, fract,
        trunc, is_finite, is_infinite,
    ], ();
    float, [log], (base: f64);
);

auto_impl_map_viewmut!(
    in1, [zscore], (min_periods: usize, stable: bool, axis: i32, par: bool);
    in1, [winsorize], (method: WinsorizeMethod, method_params: Option<f64>, stable: bool, axis: i32, par: bool);
);
