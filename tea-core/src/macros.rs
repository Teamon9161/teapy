// /// Define a 1d reduce function that can only be applied to a 1d array, then auto define
// /// a function so that we can apply this function to a `ndarray`.
// #[macro_export]
// macro_rules! impl_reduce_nd {
//     (
//         $func: ident,
//         $(#[$meta:meta])*
//         pub fn $func_1d:ident $(<$($t: ident),*>)? (&$self:ident $(, $p:ident: $p_ty:ty)* $(,)?) -> $otype:ty
//         // {$($generic:ident: $bound:path $(; $other_bnd:path)* $(,)?)*} $(,)?
//         // where $($where_pred:tt)+
//         $( where $($where_pred:tt)+ )?
//         $body: block
//     ) => {
//         impl<T: Clone, S> ArrBase<S, Ix1>
//         where
//             S: Data<Elem = T>,
//         {
//             // implement function on a given axis
//             $(#[$meta])*
//             pub fn $func_1d $(<$($t),*>)? (&$self $(, $p: $p_ty)*) -> $otype
//             $(where $($where_pred)*)?
//                 // $($generic: $bound $(+ $other_bnd)*,)*
//             $body
//         }
//         impl<T: Clone + Send + Sync, S, D> ArrBase<S, D>
//         where
//             S: Data<Elem = T>,
//             D: Dimension,
//         {
//              // make 1d function can be used in ndarray
//             $(#[$meta])*
//             pub fn $func $(<$($t),*>)? (&$self $(, $p: $p_ty)*, axis: i32, par: bool) -> ArrD<$otype>
//             where
//                 D: RemoveAxis,
//                 $($($where_pred)*)?
//                 // $($generic: $bound $(+ $other_bnd)*,)*
//             {
//                 let axis = $self.norm_axis(axis);
//                 if $self.is_empty() || $self.len_of(axis) == 0 {
//                     return Arr1::from_vec(vec![]).to_dimd();
//                     // return Arr1::from_vec(vec![]).to_dim::<D::Smaller>().unwrap();
//                 }
//                 if $self.ndim() == 1 {
//                     return ndarray::arr0($self.view().to_dim1().unwrap().$func_1d($($p),*)).wrap().to_dimd();
//                 }
//                 if !par {
//                     Zip::from($self.lanes(axis)).map_collect(move |lane| lane.wrap().$func_1d($($p.clone()),*)).wrap().to_dimd()
//                 } else {
//                     Zip::from($self.lanes(axis)).par_map_collect(move |lane| lane.wrap().$func_1d($($p.clone()),*)).wrap().to_dimd()
//                 }
//             }
//         }
//     };
// }
// // pub(super) use impl_reduce_nd;

// #[macro_export]
// /// Define a 1d map function that can only be applied to a 1d array, then auto define
// /// a function so that we can apply this function to a `ndarray`.
// macro_rules! impl_map_nd {
//     (
//         $func: ident,
//         $(#[$meta:meta])*
//         pub fn $func_1d:ident $(<S2 $(, $t: ident)*>)? (&$self:ident, $out:ident : &mut ArrBase<S2, D> $(, $p:ident: $p_ty:ty)* $(,)?) -> $otype:ty
//         {where $($generic:ident: $bound:path $(; $other_bnd:path)* $(,)?)*}
//         $body: tt
//     ) => {
//         impl<T: Clone, S> ArrBase<S, Ix1>
//         where
//             S: Data<Elem = T>,
//         {
//             $(#[$meta])*
//             pub fn $func_1d $(<S2 $(,$t)*>)? (&$self, $out: &mut ArrBase<S2, Ix1> $(, $p: $p_ty)*)
//             where
//                 S2: DataMut<Elem = MaybeUninit<$otype>>,
//                 $($generic: $bound $(+ $other_bnd)*,)*
//             $body
//         }
//         impl<T, S, D> ArrBase<S, D>
//         where
//             T: Default + Clone,
//             S: Data<Elem = T>,
//             D: Dimension,
//         {
//             $(#[$meta])*
//             pub fn $func $(<$($t),*>)? (&$self $(, $p: $p_ty)*, axis: i32, par: bool) -> Arr<$otype, D>
//             where
//                 $($generic: $bound $(+ $other_bnd)*,)*
//             {
//                 let axis = $self.norm_axis(axis);
//                 let f_flag = !$self.is_standard_layout();
//                 let shape = $self.raw_dim().into_shape().set_f(f_flag);
//                 let mut out = Arr::<$otype, D>::uninit(shape);
//                 let mut out_wr = out.view_mut();
//                 $self.apply_along_axis(&mut out_wr, axis, par, move |x_1d, mut out_1d| {
//                     x_1d.$func_1d(&mut out_1d $(,$p.clone())*)
//                 });
//                 unsafe{out.assume_init()}
//             }
//         }
//     };
// }
// // #[cfg(feature = "arr_func")]
// // pub(super) use impl_map_nd;

// #[macro_export]
// /// Define a 1d function that can only be applied to two 1d arrays, then auto define
// /// a function so that we can apply this function to two `ndarray`.
// macro_rules! impl_reduce2_nd {
//     (
//         $func: ident,
//         $(#[$meta:meta])*
//         pub fn $func_1d:ident $(<S2 $(,$t: ident)*>)? (&$self:ident, $other:ident: &ArrBase<S2, Ix1> $(, $p:ident: $p_ty:ty)* $(,)?) -> $otype:ty
//         {where $($generic:ident: $bound:path $(; $other_bnd:path)* $(,)?)*}
//         $body: tt
//     ) => {
//         impl<T: Clone, S> ArrBase<S, Ix1>
//         where
//             S: Data<Elem = T>,
//         {
//             $(#[$meta])*
//             pub fn $func_1d $(<S2 $(,$t)*>)? (&$self, $other: &ArrBase<S2, Ix1> $(, $p: $p_ty)*) -> $otype
//             where
//                 $($generic: $bound $(+ $other_bnd)*,)*
//             $body
//         }
//         impl<T: Clone, S, D> ArrBase<S, D>
//         where
//             S: Data<Elem = T>,
//             D: Dimension,
//         {
//             $(#[$meta])*
//             pub fn $func $(<S2, D2 $(,$t)*>)? (&$self, $other: &ArrBase<S2, D2> $(, $p: $p_ty)*, axis: i32, par: bool) -> Arr<$otype, <<D as DimMax<D2>>::Output as Dimension>::Smaller>
//             where
//                 D: DimMax<D2> + RemoveAxis,
//                 D2: Dimension,
//                 $($generic: $bound $(+ $other_bnd)*,)*
//             {
//                 let (lhs, rhs) = if $self.ndim() == $other.ndim() && $self.shape() == $other.shape() {
//                     let lhs = $self.view().to_dim::<<D as DimMax<D2>>::Output>().unwrap();
//                     let rhs = $other.view().to_dim::<<D as DimMax<D2>>::Output>().unwrap();
//                     (lhs, rhs)
//                 } else {
//                     $self.broadcast_with(&$other).unwrap()
//                 };

//                 let axis = lhs.norm_axis(axis);
//                 if lhs.is_empty() || lhs.len_of(axis) == 0 {
//                     return Arr1::from_vec(vec![]).to_dim::<<<D as DimMax<D2>>::Output as Dimension>::Smaller>().unwrap();
//                 }
//                 if lhs.ndim() == 1 {
//                     return ndarray::arr0(lhs.to_dim1().unwrap().$func_1d(&rhs.to_dim1().unwrap(), $($p),*)).wrap().to_dim::<<<D as DimMax<D2>>::Output as Dimension>::Smaller>().unwrap();
//                 }
//                 if !par {
//                     Zip::from(lhs.lanes(axis)).and(rhs.lanes(axis)).map_collect(|lane1, lane2| lane1.wrap().$func_1d(&lane2.wrap() $(,$p.clone())*)).into()
//                 } else {
//                     Zip::from(lhs.lanes(axis)).and(rhs.lanes(axis)).par_map_collect(|lane1, lane2| lane1.wrap().$func_1d(&lane2.wrap() $(,$p.clone())*)).into()
//                 }
//             }
//         }
//     };
// }
// // pub(super) use impl_reduce2_nd;

// #[macro_export]
// /// Define a 1d map function that can only be applied to two 1d arrays, then auto define
// /// a function so that we can apply this function to two `ndarray`.
// macro_rules! impl_map2_nd {
//     (
//         $func: ident,
//         $(#[$meta:meta])*
//         pub fn $func_1d:ident $(<S2, T2, S3 $(, $t: ident)*>)? (&$self:ident, $other:ident: &ArrBase<S2, D>, $out:ident : &mut ArrBase<S3, D> $(, $p:ident: $p_ty:ty)* $(,)?) -> $otype:ty
//         {where $($generic:ident: $bound:path $(; $other_bnd:path)* $(,)?)*}
//         $body: tt
//     ) => {
//         impl<T: Clone, S> ArrBase<S, Ix1>
//         where
//             S: Data<Elem = T>,
//         {
//             $(#[$meta])*
//             pub fn $func_1d $(<S2, T2, S3 $(,$t)*>)? (&$self, $other: &ArrBase<S2, Ix1>, $out: &mut ArrBase<S3, Ix1> $(, $p: $p_ty)*)
//             where
//                 S2: Data<Elem = T2>,
//                 S3: DataMut<Elem = MaybeUninit<$otype>>,
//                 $($generic: $bound $(+ $other_bnd)*,)*
//             $body
//         }

//         impl<T, S, D> ArrBase<S, D>
//         where
//             T: Default + Clone,
//             S: Data<Elem = T>,
//             D: Dimension,
//         {
//             $(#[$meta])*
//             pub fn $func $(<S2, D2, T2 $(,$t)*>)? (&$self, $other: &ArrBase<S2, D2> $(, $p: $p_ty)*, axis: i32, par: bool) -> Arr<$otype, <D as DimMax<D2>>::Output>
//             where
//                 S2: Data<Elem = T2>,
//                 D: DimMax<D2>,
//                 D2: Dimension,
//                 $($generic: $bound $(+ $other_bnd)*,)*
//             {
//                 let f_flag = !$self.is_standard_layout();
//                 let (lhs, rhs) = if $self.ndim() == $other.ndim() && $self.shape() == $other.shape() {
//                     let lhs = $self.view().to_dim::<<D as DimMax<D2>>::Output>().unwrap();
//                     let rhs = $other.view().to_dim::<<D as DimMax<D2>>::Output>().unwrap();
//                     (lhs, rhs)
//                 } else {
//                     $self.broadcast_with(&$other).unwrap()
//                 };
//                 let axis = lhs.norm_axis(axis);
//                 let shape = lhs.raw_dim().into_shape().set_f(f_flag);
//                 // let mut out = Arr::<$otype, <D as DimMax<D2>>::Output>::default(shape);
//                 let mut out = Arr::<$otype, <D as DimMax<D2>>::Output>::uninit(shape);
//                 let mut out_wr = out.view_mut();
//                 lhs.apply_along_axis_with(rhs, &mut out_wr, axis, par, |x_1d, y_1d, mut out_1d| {
//                     x_1d.$func_1d(&y_1d, &mut out_1d $(,$p.clone())*)
//                 });
//                 unsafe{out.assume_init()}
//             }
//         }
//     };
// }
// // pub(super) use impl_map2_nd;

// #[macro_export]
// /// Define a 1d map inplace function that can only be applied to a 1d array, then auto define
// /// a function so that we can apply this function to a `ndarray`.
// macro_rules! impl_map_inplace_nd {
//     (
//         $func: ident,
//         $(#[$meta:meta])*
//         pub fn $func_1d:ident $(<$($t: ident),*>)? (&mut $self:ident $(, $p:ident: $p_ty:ty)* $(,)?)
//         {where $($generic:ident: $bound:path $(; $other_bnd:path)* $(,)?)*}
//         $body: tt
//     ) => {
//         impl<T: Clone, S> ArrBase<S, Ix1>
//         where
//             S: DataMut<Elem = T>,
//         {
//             $(#[$meta])*
//             pub fn $func_1d $(<$($t),*>)? (&mut $self $(, $p: $p_ty)*)
//             where
//                 $($generic: $bound $(+ $other_bnd)*,)*
//             $body
//         }
//         impl<T: Clone, S, D> ArrBase<S, D>
//         where
//             S: DataMut<Elem = T>,
//             D: Dimension,
//         {
//             $(#[$meta])*
//             pub fn $func $(<$($t),*>)? (&mut $self $(, $p: $p_ty)*, axis: i32, par: bool)
//             where
//                 $($generic: $bound $(+ $other_bnd)*,)*
//             {
//                 let axis = $self.norm_axis(axis);
//                 if !par {
//                     Zip::from($self.lanes_mut(axis)).for_each(|lane| lane.wrap().$func_1d($($p.clone()),*)).into()
//                 } else {
//                     Zip::from($self.lanes_mut(axis)).par_for_each(|lane| lane.wrap().$func_1d($($p.clone()),*)).into()
//                 }
//             }
//         }
//     };
// }
// // pub(super) use impl_map_inplace_nd;

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
