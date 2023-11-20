/// Define a 1d reduce function that can only be applied to a 1d array, then auto define
/// a function so that we can apply this function to a `ndarray`.
#[macro_export]
macro_rules! impl_reduce_nd {
    (
        $func: ident,
        $(#[$meta:meta])*
        pub fn $func_1d:ident $(<$($t: ident),*>)? (&$self:ident $(, $p:ident: $p_ty:ty)* $(,)?) -> $otype:ty
        {$($generic:ident: $bound:path $(; $other_bnd:path)* $(,)?)*} $(,)?
        $body: tt
    ) => {
        impl<T: Clone, S> ArrBase<S, Ix1>
        where
            S: Data<Elem = T>,
        {
            // implement function on a given axis
            $(#[$meta])*
            pub fn $func_1d $(<$($t),*>)? (&$self $(, $p: $p_ty)*) -> $otype
            where
                $($generic: $bound $(+ $other_bnd)*,)*
            $body
        }
        impl<T: Clone, S, D> ArrBase<S, D>
        where
            S: Data<Elem = T>,
            D: Dimension,
        {
             // make 1d function can be used in ndarray
            $(#[$meta])*
            pub fn $func $(<$($t),*>)? (&$self $(, $p: $p_ty)*, axis: i32, par: bool) -> ArrD<$otype>
            where
                D: RemoveAxis,
                $($generic: $bound $(+ $other_bnd)*,)*
            {
                let axis = $self.norm_axis(axis);
                if $self.is_empty() || $self.len_of(axis) == 0 {
                    return Arr1::from_vec(vec![]).to_dimd();
                    // return Arr1::from_vec(vec![]).to_dim::<D::Smaller>().unwrap();
                }
                if $self.ndim() == 1 {
                    return ndarray::arr0($self.view().to_dim1().unwrap().$func_1d($($p),*)).wrap().to_dimd();
                }
                if !par {
                    Zip::from($self.lanes(axis)).map_collect(move |lane| lane.wrap().$func_1d($($p.clone()),*)).wrap().to_dimd()
                } else {
                    Zip::from($self.lanes(axis)).par_map_collect(move |lane| lane.wrap().$func_1d($($p.clone()),*)).wrap().to_dimd()
                }
            }
        }
    };
}
// pub(super) use impl_reduce_nd;

#[macro_export]
/// Define a 1d map function that can only be applied to a 1d array, then auto define
/// a function so that we can apply this function to a `ndarray`.
macro_rules! impl_map_nd {
    (
        $func: ident,
        $(#[$meta:meta])*
        pub fn $func_1d:ident $(<S2 $(, $t: ident)*>)? (&$self:ident, $out:ident : &mut ArrBase<S2, D> $(, $p:ident: $p_ty:ty)* $(,)?) -> $otype:ty
        {where $($generic:ident: $bound:path $(; $other_bnd:path)* $(,)?)*}
        $body: tt
    ) => {
        impl<T: Clone, S> ArrBase<S, Ix1>
        where
            S: Data<Elem = T>,
        {
            $(#[$meta])*
            pub fn $func_1d $(<S2 $(,$t)*>)? (&$self, $out: &mut ArrBase<S2, Ix1> $(, $p: $p_ty)*)
            where
                S2: DataMut<Elem = MaybeUninit<$otype>>,
                $($generic: $bound $(+ $other_bnd)*,)*
            $body
        }
        impl<T, S, D> ArrBase<S, D>
        where
            T: Default + Clone,
            S: Data<Elem = T>,
            D: Dimension,
        {
            $(#[$meta])*
            pub fn $func $(<$($t),*>)? (&$self $(, $p: $p_ty)*, axis: i32, par: bool) -> Arr<$otype, D>
            where
                $($generic: $bound $(+ $other_bnd)*,)*
            {
                let axis = $self.norm_axis(axis);
                let f_flag = !$self.is_standard_layout();
                let shape = $self.raw_dim().into_shape().set_f(f_flag);
                let mut out = Arr::<$otype, D>::uninit(shape);
                let mut out_wr = out.view_mut();
                $self.apply_along_axis(&mut out_wr, axis, par, move |x_1d, mut out_1d| {
                    x_1d.$func_1d(&mut out_1d $(,$p.clone())*)
                });
                unsafe{out.assume_init()}
            }
        }
    };
}
// #[cfg(feature = "arr_func")]
// pub(super) use impl_map_nd;

#[macro_export]
/// Define a 1d function that can only be applied to two 1d arrays, then auto define
/// a function so that we can apply this function to two `ndarray`.
macro_rules! impl_reduce2_nd {
    (
        $func: ident,
        $(#[$meta:meta])*
        pub fn $func_1d:ident $(<S2 $(,$t: ident)*>)? (&$self:ident, $other:ident: &ArrBase<S2, Ix1> $(, $p:ident: $p_ty:ty)* $(,)?) -> $otype:ty
        {where $($generic:ident: $bound:path $(; $other_bnd:path)* $(,)?)*}
        $body: tt
    ) => {
        impl<T: Clone, S> ArrBase<S, Ix1>
        where
            S: Data<Elem = T>,
        {
            $(#[$meta])*
            pub fn $func_1d $(<S2 $(,$t)*>)? (&$self, $other: &ArrBase<S2, Ix1> $(, $p: $p_ty)*) -> $otype
            where
                $($generic: $bound $(+ $other_bnd)*,)*
            $body
        }
        impl<T: Clone, S, D> ArrBase<S, D>
        where
            S: Data<Elem = T>,
            D: Dimension,
        {
            $(#[$meta])*
            pub fn $func $(<S2, D2 $(,$t)*>)? (&$self, $other: &ArrBase<S2, D2> $(, $p: $p_ty)*, axis: i32, par: bool) -> Arr<$otype, <<D as DimMax<D2>>::Output as Dimension>::Smaller>
            where
                D: DimMax<D2> + RemoveAxis,
                D2: Dimension,
                $($generic: $bound $(+ $other_bnd)*,)*
            {
                let (lhs, rhs) = if $self.ndim() == $other.ndim() && $self.shape() == $other.shape() {
                    let lhs = $self.view().to_dim::<<D as DimMax<D2>>::Output>().unwrap();
                    let rhs = $other.view().to_dim::<<D as DimMax<D2>>::Output>().unwrap();
                    (lhs, rhs)
                } else {
                    $self.broadcast_with(&$other).unwrap()
                };

                let axis = lhs.norm_axis(axis);
                if lhs.is_empty() || lhs.len_of(axis) == 0 {
                    return Arr1::from_vec(vec![]).to_dim::<<<D as DimMax<D2>>::Output as Dimension>::Smaller>().unwrap();
                }
                if lhs.ndim() == 1 {
                    return ndarray::arr0(lhs.to_dim1().unwrap().$func_1d(&rhs.to_dim1().unwrap(), $($p),*)).wrap().to_dim::<<<D as DimMax<D2>>::Output as Dimension>::Smaller>().unwrap();
                }
                if !par {
                    Zip::from(lhs.lanes(axis)).and(rhs.lanes(axis)).map_collect(|lane1, lane2| lane1.wrap().$func_1d(&lane2.wrap() $(,$p.clone())*)).into()
                } else {
                    Zip::from(lhs.lanes(axis)).and(rhs.lanes(axis)).par_map_collect(|lane1, lane2| lane1.wrap().$func_1d(&lane2.wrap() $(,$p.clone())*)).into()
                }
            }
        }
    };
}
// pub(super) use impl_reduce2_nd;

#[macro_export]
/// Define a 1d map function that can only be applied to two 1d arrays, then auto define
/// a function so that we can apply this function to two `ndarray`.
macro_rules! impl_map2_nd {
    (
        $func: ident,
        $(#[$meta:meta])*
        pub fn $func_1d:ident $(<S2, T2, S3 $(, $t: ident)*>)? (&$self:ident, $other:ident: &ArrBase<S2, D>, $out:ident : &mut ArrBase<S3, D> $(, $p:ident: $p_ty:ty)* $(,)?) -> $otype:ty
        {where $($generic:ident: $bound:path $(; $other_bnd:path)* $(,)?)*}
        $body: tt
    ) => {
        impl<T: Clone, S> ArrBase<S, Ix1>
        where
            S: Data<Elem = T>,
        {
            $(#[$meta])*
            pub fn $func_1d $(<S2, T2, S3 $(,$t)*>)? (&$self, $other: &ArrBase<S2, Ix1>, $out: &mut ArrBase<S3, Ix1> $(, $p: $p_ty)*)
            where
                S2: Data<Elem = T2>,
                S3: DataMut<Elem = MaybeUninit<$otype>>,
                $($generic: $bound $(+ $other_bnd)*,)*
            $body
        }

        impl<T, S, D> ArrBase<S, D>
        where
            T: Default + Clone,
            S: Data<Elem = T>,
            D: Dimension,
        {
            $(#[$meta])*
            pub fn $func $(<S2, D2, T2 $(,$t)*>)? (&$self, $other: &ArrBase<S2, D2> $(, $p: $p_ty)*, axis: i32, par: bool) -> Arr<$otype, <D as DimMax<D2>>::Output>
            where
                S2: Data<Elem = T2>,
                D: DimMax<D2>,
                D2: Dimension,
                $($generic: $bound $(+ $other_bnd)*,)*
            {
                let f_flag = !$self.is_standard_layout();
                let (lhs, rhs) = if $self.ndim() == $other.ndim() && $self.shape() == $other.shape() {
                    let lhs = $self.view().to_dim::<<D as DimMax<D2>>::Output>().unwrap();
                    let rhs = $other.view().to_dim::<<D as DimMax<D2>>::Output>().unwrap();
                    (lhs, rhs)
                } else {
                    $self.broadcast_with(&$other).unwrap()
                };
                let axis = lhs.norm_axis(axis);
                let shape = lhs.raw_dim().into_shape().set_f(f_flag);
                // let mut out = Arr::<$otype, <D as DimMax<D2>>::Output>::default(shape);
                let mut out = Arr::<$otype, <D as DimMax<D2>>::Output>::uninit(shape);
                let mut out_wr = out.view_mut();
                lhs.apply_along_axis_with(rhs, &mut out_wr, axis, par, |x_1d, y_1d, mut out_1d| {
                    x_1d.$func_1d(&y_1d, &mut out_1d $(,$p.clone())*)
                });
                unsafe{out.assume_init()}
            }
        }
    };
}
// pub(super) use impl_map2_nd;

#[macro_export]
/// Define a 1d map inplace function that can only be applied to a 1d array, then auto define
/// a function so that we can apply this function to a `ndarray`.
macro_rules! impl_map_inplace_nd {
    (
        $func: ident,
        $(#[$meta:meta])*
        pub fn $func_1d:ident $(<$($t: ident),*>)? (&mut $self:ident $(, $p:ident: $p_ty:ty)* $(,)?)
        {where $($generic:ident: $bound:path $(; $other_bnd:path)* $(,)?)*}
        $body: tt
    ) => {
        impl<T: Clone, S> ArrBase<S, Ix1>
        where
            S: DataMut<Elem = T>,
        {
            $(#[$meta])*
            pub fn $func_1d $(<$($t),*>)? (&mut $self $(, $p: $p_ty)*)
            where
                $($generic: $bound $(+ $other_bnd)*,)*
            $body
        }
        impl<T: Clone, S, D> ArrBase<S, D>
        where
            S: DataMut<Elem = T>,
            D: Dimension,
        {
            $(#[$meta])*
            pub fn $func $(<$($t),*>)? (&mut $self $(, $p: $p_ty)*, axis: i32, par: bool)
            where
                $($generic: $bound $(+ $other_bnd)*,)*
            {
                let axis = $self.norm_axis(axis);
                if !par {
                    Zip::from($self.lanes_mut(axis)).for_each(|lane| lane.wrap().$func_1d($($p.clone()),*)).into()
                } else {
                    Zip::from($self.lanes_mut(axis)).par_for_each(|lane| lane.wrap().$func_1d($($p.clone()),*)).into()
                }
            }
        }
    };
}
// pub(super) use impl_map_inplace_nd;
