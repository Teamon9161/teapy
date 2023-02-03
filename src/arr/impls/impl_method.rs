use super::super::{Arr, ArrBase, ArrView, Data, RawData, WrapNdarray};
use ndarray::{Axis, DataMut, DimMax, Dimension, ErrorKind, IntoDimension, ShapeError, Zip};
// use std::fmt::Debug;

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
pub fn co_broadcast<D1, D2, Output>(shape1: &D1, shape2: &D2) -> Result<Output, ShapeError>
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
    for (out, s2) in izip!(&mut out.slice_mut()[k..], shape2.slice()) {
        if *out != *s2 {
            if *out == 1 {
                *out = *s2
            } else if *s2 != 1 {
                return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
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
    ) -> Result<
        (
            ArrView<'a, A, DimMaxOf<D, E>>,
            ArrView<'b, B, DimMaxOf<D, E>>,
        ),
        ShapeError,
    >
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
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        };
        let view2 = if shape.slice() == other.raw_dim().slice() {
            other.view().to_dim::<<D as DimMax<E>>::Output>().unwrap()
        } else if let Some(view2) = other.broadcast(shape) {
            view2
        } else {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        };
        Ok((view1, view2))
    }
}

impl<A, S, D: Dimension> ArrBase<S, D>
where
    S: Data<Elem = A>,
{
    pub fn put_mask<S2, S3, D2, D3>(
        &mut self,
        mask: &ArrBase<S2, D2>,
        value: &ArrBase<S3, D3>,
        axis: usize,
        par: bool,
    ) where
        A: Clone + Send + Sync,
        D2: Dimension,
        D3: Dimension,
        S: DataMut<Elem = A>,
        S2: Data<Elem = bool>,
        S3: Data<Elem = A>,
    {
        let value = if self.ndim() == value.ndim() && self.shape() == value.shape() {
            value.view().to_dim::<D>().unwrap()
        } else if let Some(value) = value.broadcast(self.raw_dim()) {
            value
        } else {
            // the number of value array's elements are equal to the number of true values in mask
            let mask = mask
                .view()
                .to_dim1()
                .expect("mask should be dim1 when set value to masked data");
            let value = value
                .view()
                .to_dim1()
                .expect("value should be dim1 when set value to masked data");
            let true_num = mask.count_v_1d(true) as usize;
            assert_eq!(
                true_num,
                value.len_of(Axis(axis)),
                "number of value are not equal to number of true mask"
            );
            let ndim = self.ndim();
            return if par && (ndim > 1) {
                Zip::from(self.lanes_mut(Axis(axis)))
                    .par_for_each(|x_1d| x_1d.wrap().put_mask_1d(&mask, &value));
            } else {
                Zip::from(self.lanes_mut(Axis(axis)))
                    .for_each(|x_1d| x_1d.wrap().put_mask_1d(&mask, &value));
            };
        };
        let mask = if self.ndim() == mask.ndim() && self.shape() == mask.shape() {
            mask.view().to_dim::<D>().unwrap()
        } else {
            mask.broadcast(self.raw_dim()).unwrap()
        };
        if !par {
            Zip::from(&mut self.0)
                .and(&mask.0)
                .and(&value.0)
                .for_each(|a, m, v| {
                    if *m {
                        *a = v.clone()
                    }
                })
        } else {
            Zip::from(&mut self.0)
                .and(&mask.0)
                .and(&value.0)
                .par_for_each(|a, m, v| {
                    if *m {
                        *a = v.clone()
                    }
                })
        }
    }

    pub fn where_<S2, S3, D2, D3>(
        &self,
        mask: &ArrBase<S2, D2>,
        value: &ArrBase<S3, D3>,
        par: bool,
    ) -> Arr<A, <D as DimMax<D3>>::Output>
    where
        A: Clone + Send + Sync,
        D: DimMax<D3>,
        D2: Dimension,
        D3: Dimension,
        S2: Data<Elem = bool>,
        S3: Data<Elem = A>,
    {
        let (lhs, rhs) = if self.ndim() == value.ndim() && self.shape() == value.shape() {
            let lhs = self.view().to_dim::<<D as DimMax<D3>>::Output>().unwrap();
            let rhs = value.view().to_dim::<<D as DimMax<D3>>::Output>().unwrap();
            (lhs, rhs)
        } else {
            self.broadcast_with(value).unwrap()
        };

        let mask = if lhs.ndim() == mask.ndim() && lhs.shape() == mask.shape() {
            mask.view().to_dim::<<D as DimMax<D3>>::Output>().unwrap()
        } else {
            mask.broadcast(lhs.raw_dim()).unwrap()
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