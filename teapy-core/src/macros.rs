#[macro_export]
macro_rules! match_enum {
    // select the match arm
    ($enum: ident, $exprs: expr, $e: ident, $body: tt, $($(#[$meta: meta])? $arm: ident),* $(,)?) => {
        match $exprs {
            $($(#[$meta])? $enum::$arm($e) => $body,)*
            _ => Err(terr!("Not supported arm for enum {:?}", stringify!($enum)))
        }
    };

    ($enum: ident, $exprs: expr; $($rest: tt)*) => {
        $crate::match_enum!(@($enum, $exprs; $($rest)*))
    };

    // match all arm
    (@($enum: ident, $exprs: expr; dtype_all ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs;
                (
                    Normal | String | VecUsize | Str
                    | Object
                    | #[cfg(feature="time")] DateTime
                    | #[cfg(feature="time")] TimeDelta
                )
                ($e) => $body,
            $($rest)*)
            $($all_arms)*
        )
    };


    // match dynamic & castable arm
    (@($enum: ident, $exprs: expr; Cast ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs;
                (Normal | String | Object | #[cfg(feature="time")] TimeDelta)($e) => $body,
            $($rest)*)
            $($all_arms)*
        )
    };


    // match normal dtype, no str, object, time, vecusize
    (@($enum: ident, $exprs: expr; Normal ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs; (Numeric | BoolLike)($e) => $body, $($rest)*)
            $($all_arms)*
        )
    };

    // match dtype that support Dynamic(currently str doesn't support)
    (@($enum: ident, $exprs: expr; Dynamic ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs;
                (IntoPy | TimeRelated)($e) => $body,
            $($rest)*)
            $($all_arms)*
        )
    };

    // match dtype that support IntoPy(currently timelike doesn't support)
    (@($enum: ident, $exprs: expr; IntoPy ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs;
                (AsRefPy | Object)($e) => $body,
            $($rest)*)
            $($all_arms)*
        )
    };

    // match dtype that support AsRef Py(currently timelike doesn't support)
    (@($enum: ident, $exprs: expr; AsRefPy ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        // TODO: should support str type
        $crate::match_enum!(
            @($enum, $exprs;
                (Normal | String | VecUsize)($e) => $body,
            $($rest)*)
            $($all_arms)*
        )
    };

    // match option type support by polars
    (@($enum: ident, $exprs: expr; PlOpt ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs;
                (PlOptInt | OptFloat | OptBool)($e) => $body,
            $($rest)*)
            $($all_arms)*
        )
    };

    // match pure int support by polars
    (@($enum: ident, $exprs: expr; PlInt ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs;
                (I32 | I64 | U64)($e) => $body,
            $($rest)*)
            $($all_arms)*
        )
    };

    // match pure int like arm
    (@($enum: ident, $exprs: expr; PureInt ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs;
                (PlInt | Usize)($e) => $body,
            $($rest)*)
            $($all_arms)*
        )
    };

    // match option int support by polars
    (@($enum: ident, $exprs: expr; PlOptInt ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs;
                (OptI32 | OptI64)($e) => $body,
            $($rest)*)
            $($all_arms)*
        )
    };

    // match option int support by polars
    (@($enum: ident, $exprs: expr; OptInt ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs;
                (PlOptInt | OptUsize)($e) => $body,
            $($rest)*)
            $($all_arms)*
        )
    };

    // match int like arm (pure_int + OptUsize)
    (@($enum: ident, $exprs: expr; Int ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs;
                (PureInt | OptInt)($e) => $body,
            $($rest)*)
            $($all_arms)*
        )
    };

    // match bool like arm
    (@($enum: ident, $exprs: expr; BoolLike ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs;
                (Bool | U8 | OptBool)($e) => $body,
            $($rest)*)
            $($all_arms)*
        )
    };

    // match option float like arm
    (@($enum: ident, $exprs: expr; OptFloat ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs; (OptF32 | OptF64)($e) => $body, $($rest)*)
            $($all_arms)*
        )
    };

    // match float like arm
    (@($enum: ident, $exprs: expr; PureFloat ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs; (F32 | F64)($e) => $body, $($rest)*)
            $($all_arms)*
        )
    };

    // match float like arm
    (@($enum: ident, $exprs: expr; Float ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs; (PureFloat | OptFloat)($e) => $body, $($rest)*)
            $($all_arms)*
        )
    };

    // match time like arm
    (@($enum: ident, $exprs: expr; Time ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs;
            (
                #[cfg(feature="time")] DateTimeMs
                | #[cfg(feature="time")] DateTimeUs
                | #[cfg(feature="time")] DateTimeNs
            )
            ($e) => $body,
            $($rest)*)
            $($all_arms)*
        )
    };

    // match time related arm
    (@($enum: ident, $exprs: expr; TimeRelated ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs;
            (
                Time
                | #[cfg(feature="time")] TimeDelta
            )
            ($e) => $body,
            $($rest)*)
            $($all_arms)*
        )
    };

    // match pure numeric arm (pure int + float)
    (@($enum: ident, $exprs: expr; PureNumeric ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs; (PureInt | PureFloat)($e) => $body, $($rest)*)
            $($all_arms)*
        )
    };

    // match numeric( int + float ) arm
    (@($enum: ident, $exprs: expr; Numeric ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs; (Int | Float)($e) => $body, $($rest)*)
            $($all_arms)*
        )
    };

    // match hashable dtype
    (@($enum: ident, $exprs: expr; Hash ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs; (PureInt | String | Bool | U8 | Time)($e) => $body, $($rest)*)
            $($all_arms)*
        )
    };

    // match hashable dtype
    (@($enum: ident, $exprs: expr; TpHash ($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs; (PureNumeric | String | Bool | U8 | Time)($e) => $body, $($rest)*)
            $($all_arms)*
        )
    };

    // expand | in macro (for example: (I32 | I64)(e))
    (@($enum: ident, $exprs: expr; ($($(#[$meta: meta])? $arms: ident)|+)($e: ident) => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs; $($(#[$meta])? $arms($e) => $body,)+ $($rest)*)
            $($all_arms)*
        )
    };

    // expand | in macro (for example: I32(e) | I64(e))
    (@($enum: ident, $exprs: expr; $(#[$meta: meta])? $arms: ident ($e: ident) $(| $(#[$other_meta: meta])? $other_arms: ident ($other_e: ident))+ => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs; $(#[$meta])? $arms($e) => $body, $($(#[$other_meta])? $other_arms ($other_e) => $body,)* $($rest)*)
            $($all_arms)*
        )
    };


    // match one arm, note that this rule should be the last one
    (@($enum: ident, $exprs: expr; $($(#[$meta: meta])? $arms: ident ($e: ident))|+ => $body: expr, $($rest: tt)*) $($all_arms: tt)* ) => {
        $crate::match_enum!(
            @($enum, $exprs; $($rest)*)
            $($all_arms)*
            $($(#[$meta])? $enum::$arms($e) => $body,)*
        )
    };

    // No more match arms, produce final output
    (@($enum: ident, $exprs: expr; $(,)?) $($all_arms: tt)*) => {
        {
            // use $enum::*;
            match $exprs {
                $($all_arms)*
                _ => Err(terr!("Not supported arm for enum {:?}", stringify!($enum)))
            }
        }
    };

    ($enum: ident, ($exprs1: expr, $e1: ident, $($arm1: ident),*), ($exprs2: expr, $e2: ident, $($arm2: ident),*), $body: tt) => {
        $crate::match_enum!($enum, $exprs1, $e1, {$crate::match_enum!($enum, $exprs2, $e2, $body, $($arm2),*)}, $($arm1),*)
    };
}
