use crate::prelude::{Arr, ArrBase, ArrView, TpResult, WrapNdarray};
use ndarray::{Data, DimMax, Dimension, IntoDimension, RawData, Zip};

type DimMaxOf<A, B> = <A as DimMax<B>>::Output;

/// Create an iterator running multiple iterators in lockstep.
///
/// The `izip!` iterator yields elements until any subiterator
/// returns `None`.
///
/// This is a version of the standard ``.zip()`` that's supporting more than
/// two iterators. The iterator element type is a tuple with one element
/// from each of the input iterators. Just like ``.zip()``, the iteration stops
/// when the shortest of the inputs reaches its end.
///
/// **Note:** The result of this macro is in the general case an iterator
/// composed of repeated `.zip()` and a `.map()`; it has an anonymous type.
/// The special cases of one and two arguments produce the equivalent of
/// `$a.into_iter()` and `$a.into_iter().zip($b)` respectively.
///
/// Prefer this macro `izip!()` over `multizip` for the performance benefits
/// of using the standard library `.zip()`.
///
/// ```
/// #[macro_use] extern crate itertools;
/// # fn main() {
///
/// // iterate over three sequences side-by-side
/// let mut results = [0, 0, 0, 0];
/// let inputs = [3, 7, 9, 6];
///
/// for (r, index, input) in izip!(&mut results, 0..10, &inputs) {
///     *r = index * 10 + input;
/// }
///
/// assert_eq!(results, [0 + 3, 10 + 7, 29, 36]);
/// # }
/// ```
///
/// **Note:** To enable the macros in this crate, use the `#[macro_use]`
/// attribute when importing the crate:
///
/// ```no_run
/// # #[allow(unused_imports)]
/// #[macro_use] extern crate itertools;
/// # fn main() { }
/// ```
macro_rules! izip {
    // @closure creates a tuple-flattening closure for .map() call. usage:
    // @closure partial_pattern => partial_tuple , rest , of , iterators
    // eg. izip!( @closure ((a, b), c) => (a, b, c) , dd , ee )
    ( @closure $p:pat => $tup:expr ) => {
        |$p| $tup
    };

    // The "b" identifier is a different identifier on each recursion level thanks to hygiene.
    ( @closure $p:pat => ( $($tup:tt)* ) , $_iter:expr $( , $tail:expr )* ) => {
        izip!(@closure ($p, b) => ( $($tup)*, b ) $( , $tail )*)
    };

    // unary
    ($first:expr $(,)*) => {
        IntoIterator::into_iter($first)
    };

    // binary
    ($first:expr, $second:expr $(,)*) => {
        izip!($first)
            .zip($second)
    };

    // n-ary where n > 2
    ( $first:expr $( , $rest:expr )* $(,)* ) => {
        izip!($first)
            $(
                .zip($rest)
            )*
            .map(
                izip!(@closure a => (a) $( , $rest )*)
            )
    };
}

/// Calculate the common shape for a pair of array shapes, that they can be broadcasted
/// to. Return an error if the shapes are not compatible.
///
/// Uses the [NumPy broadcasting rules]
//  (https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules).
#[allow(unused_mut)]
pub fn co_broadcast<D1, D2, Output>(shape1: &D1, shape2: &D2) -> TpResult<Output>
where
    D1: Dimension,
    D2: Dimension,
    Output: Dimension,
{
    let (k, overflow) = shape1.ndim().overflowing_sub(shape2.ndim());
    // Swap the order if d2 is longer.
    if overflow {
        return co_broadcast::<D2, D1, Output>(shape2, shape1);
    }
    // The output should be the same length as shape1.
    let mut out = Output::zeros(shape1.ndim());
    for (out, s) in izip!(out.slice_mut(), shape1.slice()) {
        *out = *s;
    }
    for (mut out, s2) in izip!(&mut out.slice_mut()[k..], shape2.slice()) {
        if *out != *s2 {
            if *out == 1 {
                *out = *s2
            } else if *s2 != 1 {
                return Err("IncompatibleShape in co_broadcast".into());
            }
        }
    }
    Ok(out)
}

impl<A, S: RawData<Elem = A>, D: Dimension> ArrBase<S, D> {
    /// Act like a larger size and/or shape array by *broadcasting*
    /// into a larger shape, if possible.
    ///
    /// Return `None` if shapes can not be broadcast together.
    ///
    /// ***Background***
    ///
    ///  * Two axes are compatible if they are equal, or one of them is 1.
    ///  * In this instance, only the axes of the smaller side (self) can be 1.
    ///
    /// Compare axes beginning with the *last* axis of each shape.
    ///
    /// For example (1, 2, 4) can be broadcast into (7, 6, 2, 4)
    /// because its axes are either equal or 1 (or missing);
    /// while (2, 2) can *not* be broadcast into (2, 4).
    ///
    /// The implementation creates a view with strides set to zero for the
    /// axes that are to be repeated.
    ///
    /// The broadcasting documentation for Numpy has more information.
    ///
    /// ```
    /// use ndarray::{aview1, aview2};
    ///
    /// assert!(
    ///     aview1(&[1., 0.]).broadcast((10, 2)).unwrap()
    ///     == aview2(&[[1., 0.]; 10])
    /// );
    /// ```
    #[inline(always)]
    pub fn broadcast<E>(&self, dim: E) -> Option<ArrView<'_, A, E::Dim>>
    where
        E: IntoDimension,
        S: Data,
    {
        self.0.broadcast(dim).map(|arr| arr.wrap())
    }
    /// For two arrays or views, find their common shape if possible and
    /// broadcast them as array views into that shape.
    ///
    /// Return `ShapeError` if their shapes can not be broadcast together.
    #[allow(clippy::type_complexity)]
    pub fn broadcast_with<'a, 'b, B, S2, E>(
        &'a self,
        other: &'b ArrBase<S2, E>,
    ) -> TpResult<(
        ArrView<'a, A, DimMaxOf<D, E>>,
        ArrView<'b, B, DimMaxOf<D, E>>,
    )>
    where
        S: Data<Elem = A>,
        S2: Data<Elem = B>,
        D: Dimension + DimMax<E>,
        E: Dimension,
    {
        let shape =
            co_broadcast::<D, E, <D as DimMax<E>>::Output>(&self.raw_dim(), &other.raw_dim())?;
        let view1 = if shape.slice() == self.raw_dim().slice() {
            self.view().to_dim::<<D as DimMax<E>>::Output>().unwrap()
        } else if let Some(view1) = self.broadcast(shape.clone()) {
            view1
        } else {
            return Err("IncompatibleShape in broadcast_with".into());
        };
        let view2 = if shape.slice() == other.raw_dim().slice() {
            other.view().to_dim::<<D as DimMax<E>>::Output>().unwrap()
        } else if let Some(view2) = other.broadcast(shape) {
            view2
        } else {
            return Err("IncompatibleShape in broadcast_with".into());
        };
        Ok((view1, view2))
    }
}

impl<A, S, D: Dimension> ArrBase<S, D>
where
    S: Data<Elem = A>,
{
    pub fn where_<S2, S3, D2, D3>(
        &self,
        mask: &ArrBase<S2, D2>,
        value: &ArrBase<S3, D3>,
        par: bool,
    ) -> Arr<A, DimMaxOf<D2, DimMaxOf<D, D3>>>
    where
        A: Clone + Send + Sync,
        D: DimMax<D3>,
        D2: Dimension + DimMax<DimMaxOf<D, D3>>,
        D3: Dimension,
        S2: Data<Elem = bool>,
        S3: Data<Elem = A>,
    {
        // <D2 as DimMax<<D as DimMax<D3>>::Output>>::Output
        let (lhs, rhs) = if self.ndim() == value.ndim() && self.shape() == value.shape() {
            let lhs = self.view().to_dim::<DimMaxOf<D, D3>>().unwrap();
            let rhs = value.view().to_dim::<DimMaxOf<D, D3>>().unwrap();
            (lhs, rhs)
        } else {
            self.broadcast_with(value).unwrap()
        };

        let (mask, lhs, rhs) = if lhs.ndim() == mask.ndim() && lhs.shape() == mask.shape() {
            (
                mask.view()
                    .to_dim::<DimMaxOf<D2, DimMaxOf<D, D3>>>()
                    .unwrap(),
                lhs.to_dim::<DimMaxOf<D2, DimMaxOf<D, D3>>>().unwrap(),
                rhs.to_dim::<DimMaxOf<D2, DimMaxOf<D, D3>>>().unwrap(),
            )
        } else {
            let (mask_new, lhs) = mask.broadcast_with(&lhs).unwrap();
            // let rhs = rhs.broadcast(&lhs.raw_dim()).unwrap();
            let (_, rhs) = mask.broadcast_with(&rhs).unwrap();
            (mask_new, lhs, rhs)
        };
        if !par {
            Zip::from(&lhs.0)
                .and(&mask.0)
                .and(&rhs.0)
                .map_collect(|a, m, v| if *m { a.clone() } else { v.clone() })
                .wrap()
        } else {
            Zip::from(&lhs.0)
                .and(&mask.0)
                .and(&rhs.0)
                .par_map_collect(|a, m, v| if *m { a.clone() } else { v.clone() })
                .wrap()
        }
    }
}
