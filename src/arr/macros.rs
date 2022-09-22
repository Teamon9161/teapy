/// define the compensation to use in kahan summation.
macro_rules! define_c {
    ($($c: ident),*) => {
        $(let $c = &mut 0.;)*
    }
}
pub(super) use define_c;

macro_rules! define_n {
    ($($n: ident),*) => {
        $(let $n = &mut 0usize;)*
    }
}
pub(super) use define_n;

/// Define a 1d reduce function that can only be applied to a 1d array, then auto define
/// a function so that we can apply this function to a `ndarray`.
macro_rules! impl_reduce_nd {
    (
        $func: ident,
        $(#[$meta:meta])*
        pub fn $func_1d:ident $(<$($t: ident),*>)? (&$self:ident $(, $p:ident: $p_ty:ty)* $(,)?) -> $otype:ty
        {$($generic:ident: $bound:path $(; $other_bnd:path)* $(,)?)*} $(,)?
        $body: tt
    ) => {
        $(#[$meta])*
        pub fn $func_1d $(<$($t),*>)? (&$self $(, $p: $p_ty)*) -> $otype
        where
            D: Dim1,
            usize: NdIndex<D>,
            $($generic: $bound $(+ $other_bnd)*,)*
        $body

        $(#[$meta])*
        pub fn $func $(<$($t),*>)? (&$self $(, $p: $p_ty)*, axis: usize, par: bool) -> Arr<$otype, D::Smaller>
        where
            D: RemoveAxis,
            $($generic: $bound $(+ $other_bnd)*,)*
        {
            if !par {
                Zip::from($self.lanes(Axis(axis))).map_collect(|lane| ArrBase::new(lane).$func_1d($($p),*)).into()
            } else {
                Zip::from($self.lanes(Axis(axis))).par_map_collect(|lane| ArrBase::new(lane).$func_1d($($p),*)).into()
            }
        }
    };
}
pub(super) use impl_reduce_nd;

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
        $(#[$meta])*
        pub fn $func_1d $(<S2 $(,$t)*>)? (&$self, $out: &mut ArrBase<S2, D> $(, $p: $p_ty)*)
        where
            S2: DataMut<Elem = $otype>,
            D: Dim1,
            usize: NdIndex<D>,
            $($generic: $bound $(+ $other_bnd)*,)*
        $body

        $(#[$meta])*
        pub fn $func $(<$($t),*>)? (&$self $(, $p: $p_ty)*, axis: usize, par: bool) -> Arr<$otype, D>
        where
            $($generic: $bound $(+ $other_bnd)*,)*
        {
            let f_flag = !$self.is_standard_layout();
            let shape = $self.raw_dim().into_shape().set_f(f_flag);
            let mut out = Arr::<$otype, D>::zeros(shape);
            let mut out_wr = out.view_mut();
            $self.apply_along_axis(&mut out_wr, Axis(axis), par, |x_1d, out_1d| {
                x_1d.wrap().$func_1d(&mut out_1d.wrap() $(,$p)*)
            });
            out
        }
    };
}
pub(super) use impl_map_nd;

/// Define a 1d function that can only be applied to two 1d arrays, then auto define
/// a function so that we can apply this function to two `ndarray`.
macro_rules! impl_agg2_nd {
    (
        $func: ident,
        $(#[$meta:meta])*
        pub fn $func_1d:ident $(<$($t: ident),*>)? (&$self:ident, $other:ident: &$other_ty:ty $(, $p:ident: $p_ty:ty)* $(,)?) -> $otype:ty
        {where $($generic:ident: $bound:path $(; $other_bnd:path)* $(,)?)*}
        $body: tt
    ) => {
        $(#[$meta])*
        pub fn $func_1d $(<$($t),*>)? (&$self, $other: &$other_ty $(, $p: $p_ty)*) -> $otype
        where
            D: Dim1,
            usize: NdIndex<D>,
            $($generic: $bound $(+ $other_bnd)*,)*
        $body

        $(#[$meta])*
        pub fn $func $(<$($t),*>)? (&$self, $other: &$other_ty $(, $p: $p_ty)*, axis: usize, par: bool) -> Arr<$otype, D::Smaller>
        where
            D: RemoveAxis,
            $($generic: $bound $(+ $other_bnd)*,)*
        {
            let axis = Axis(axis);
            if !par {
                Zip::from($self.lanes(axis)).and($other.lanes(axis)).map_collect(|lane1, lane2| ArrBase::new(lane1).$func_1d(&ArrBase::new(lane2) $(,$p)*)).into()
            } else {
                Zip::from($self.lanes(axis)).and($other.lanes(axis)).par_map_collect(|lane1, lane2| ArrBase::new(lane1).$func_1d(&ArrBase::new(lane2) $(,$p)*)).into()
            }
        }
    };
}
pub(super) use impl_agg2_nd;

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
        $(#[$meta])*
        pub fn $func_1d $(<S2, T2, S3 $(,$t)*>)? (&$self, $other:&ArrBase<S2, D>, $out: &mut ArrBase<S3, D> $(, $p: $p_ty)*)
        where
            S2: Data<Elem = T2>,
            S3: DataMut<Elem = $otype>,
            D: Dim1,
            usize: NdIndex<D>,
            $($generic: $bound $(+ $other_bnd)*,)*
        $body

        $(#[$meta])*
        pub fn $func $(<S2, T2 $(,$t)*>)? (&$self, $other: &ArrBase<S2, D> $(, $p: $p_ty)*, axis: usize, par: bool) -> Arr<$otype, D>
        where
            S2: Data<Elem = T2>,
            $($generic: $bound $(+ $other_bnd)*,)*
        {
            let f_flag = !$self.is_standard_layout();
            let shape = $self.raw_dim().into_shape().set_f(f_flag);
            let mut out = Arr::<$otype, D>::zeros(shape);
            let mut out_wr = out.view_mut();
            $self.apply_along_axis_with($other, &mut out_wr, Axis(axis), par, |x_1d, y_1d, out_1d| {
                x_1d.wrap().$func_1d(&y_1d.wrap(), &mut out_1d.wrap() $(,$p)*)
            });
            out
        }
    };
}
pub(super) use impl_map2_nd;

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
        $(#[$meta])*
        pub fn $func_1d $(<$($t),*>)? (&mut $self $(, $p: $p_ty)*)
        where
            D: Dim1,
            usize: NdIndex<D>,
            $($generic: $bound $(+ $other_bnd)*,)*
        $body

        $(#[$meta])*
        pub fn $func $(<$($t),*>)? (&mut $self $(, $p: $p_ty)*, axis: usize, par: bool)
        where
            $($generic: $bound $(+ $other_bnd)*,)*
        {
            let axis = Axis(axis);
            if !par {
                Zip::from($self.lanes_mut(axis)).for_each(|lane| lane.wrap().$func_1d($($p),*)).into()
            } else {
                Zip::from($self.lanes_mut(axis)).par_for_each(|lane| lane.wrap().$func_1d($($p),*)).into()
            }
        }
    };
}
pub(super) use impl_map_inplace_nd;
