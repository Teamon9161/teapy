use crate::prelude::{arr0, ArrD, ArrOk, ArrViewD, ArrViewMutD, WrapNdarray};
use derive_more::From;
use ndarray::{s, Array, Axis, IxDyn, NewAxis, ShapeBuilder, SliceArg};
#[cfg(feature = "arw")]
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::marker::PhantomPinned;
use std::ops::Deref;
use std::pin::Pin;
use tevec::prelude::*;

#[derive(From)]
pub enum ArbArray<'a, T> {
    View(ArrViewD<'a, T>),
    ViewMut(ArrViewMutD<'a, T>),
    Owned(ArrD<T>),
    ViewOnBase(Pin<Box<ViewOnBase<'a, T>>>),
    #[cfg(feature = "arw")]
    ArrowChunk(Vec<Box<dyn arrow::array::Array>>),
}

#[derive(From)]
pub enum ViewBase<'a, T> {
    ArbArray(ArbArray<'a, T>),
    #[cfg(feature = "arw")]
    Arrow(Box<dyn arrow::array::Array>),
}

pub struct ViewOnBase<'a, T> {
    pub base: ViewBase<'a, T>, // ArbArray<'a, T>,
    pub view: Option<ArrViewD<'a, T>>,
    _pin: PhantomPinned,
}

impl<'a, T> Deref for ViewOnBase<'a, T> {
    type Target = ArrViewD<'a, T>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.view.as_ref().unwrap()
    }
}

impl<'a, T: Clone> Clone for ArbArray<'a, T> {
    #[inline]
    fn clone(&self) -> Self {
        self.view().to_owned().into()
    }
}

impl<'a, T> ViewOnBase<'a, T> {
    #[inline(always)]
    pub fn view(&self) -> &ArrViewD<'a, T> {
        self.view.as_ref().unwrap()
    }

    #[inline]
    pub fn new(arr: ArbArray<'a, T>, view: ArrViewD<'a, T>) -> Pin<Box<Self>> {
        let out = Self {
            base: arr.into(),
            view: Some(view),
            _pin: PhantomPinned,
        };
        Box::pin(out)
    }

    #[cfg(feature = "arw")]
    #[inline(always)]
    pub fn new_from_arrow(
        arr: Box<dyn arrow::array::Array>,
        view: ArrViewD<'a, T>,
    ) -> Pin<Box<Self>> {
        let out = Self {
            base: arr.into(),
            view: Some(view),
            _pin: PhantomPinned,
        };
        Box::pin(out)
    }
}

impl<'a, T: std::fmt::Debug> std::fmt::Debug for ArbArray<'a, T> {
    #[allow(unreachable_patterns)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            ArbArray::View(a) => write!(f, "ArrayView({a:#?})"),
            ArbArray::ViewMut(a) => write!(f, "ArrayViewMut({a:#?})"),
            ArbArray::Owned(a) => write!(f, "ArrayOwned({a:#?})"),
            ArbArray::ViewOnBase(a) => write!(f, "ViewonBase({:#?})", a.view()),
            #[cfg(feature = "arw")]
            ArbArray::ArrowChunk(ac) => write!(f, "ArrowChunk({:#?})", ac),
        }
    }
}

#[cfg(not(feature = "arw"))]
impl<'a, T: Default> Default for ArbArray<'a, T> {
    #[inline(always)]
    fn default() -> Self {
        ArbArray::Owned(Default::default())
    }
}

#[cfg(feature = "arw")]
impl<'a, T> Default for ArbArray<'a, T> {
    #[inline(always)]
    fn default() -> Self {
        ArbArray::ArrowChunk(vec![])
    }
}

macro_rules! match_arbarray {
    ($arb_array: expr, $arr: ident, $body: tt) => {
        match_arbarray!($arb_array, $arr, $body, (View, ViewMut, Owned, ViewOnBase))
    };

    ($arb_array: expr, $arr: ident, $body: tt, ($($arm: ident),*)) => {
        match $arb_array {
            $(ArbArray::$arm($arr) => $body,)*
            _ => panic!("Invalid match of ArbArray")
        }
    };
}
pub(crate) use match_arbarray;

impl<'a, T> ArbArray<'a, T> {
    #[inline]
    pub fn raw_dim(&self) -> IxDyn {
        self.view().raw_dim()
    }

    #[inline(always)]
    pub fn dtype(&self) -> DataType
    where
        T: GetDataType,
    {
        T::dtype()
    }

    #[cfg(not(feature = "arw"))]
    #[inline]
    pub fn prepare(&mut self) {}

    #[cfg(feature = "arw")]
    pub fn prepare(&mut self)
    where
        ArrOk<'a>: Cast<Self>,
    {
        if let ArbArray::ArrowChunk(_) = &self {
            let arrow_chunk = std::mem::take(self);
            if let ArbArray::ArrowChunk(arrow_chunk) = arrow_chunk {
                let arr_vec = arrow_chunk
                    .into_par_iter()
                    .map(ArrOk::from_arrow)
                    .collect::<Vec<_>>();
                let arr = ArrOk::same_dtype_concat_1d(arr_vec);
                *self = arr.cast()
            }
        }
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn ndim(&self) -> usize {
        match_arbarray!(self, a, { a.ndim() })
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn shape(&self) -> &[usize] {
        match_arbarray!(self, a, { a.shape() })
    }

    #[allow(unreachable_patterns, clippy::len_without_is_empty)]
    #[inline]
    pub fn len(&self) -> usize {
        match_arbarray!(self, a, { a.len() })
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn len_of(&self, axis: Axis) -> usize {
        match_arbarray!(self, a, { a.len_of(axis) })
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn norm_axis(&self, axis: i32) -> Axis {
        match_arbarray!(self, a, { a.norm_axis(axis) })
    }

    #[inline]
    pub fn from_scalar(scalar: T) -> Self {
        ArbArray::Owned(arr0(scalar).into_dyn())
    }

    #[inline]
    pub fn deref(&self) -> ArbArray<'_, T> {
        match &self {
            ArbArray::View(view) => view.view().into(),
            ArbArray::ViewMut(view) => view.view().into(),
            ArbArray::Owned(arr) => arr.view().into(),
            ArbArray::ViewOnBase(vb) => vb.view().view().into(),
            #[cfg(feature = "arw")]
            ArbArray::ArrowChunk(_) => {
                unreachable!("Can not deref ArrowChunk Array before rechunk")
            }
        }
    }

    #[inline]
    pub fn is_owned(&self) -> bool {
        matches!(self, ArbArray::Owned(_))
    }

    #[inline]
    pub fn is_float(&self) -> bool
    where
        T: GetDataType,
    {
        self.dtype().is_float()
    }

    #[inline]
    pub fn is_int(&self) -> bool
    where
        T: GetDataType,
    {
        self.dtype().is_int()
    }

    /// Cast to another type
    #[inline]
    pub fn cast<T2>(self) -> ArbArray<'a, T2>
    where
        T: GetDataType + Cast<T2> + Clone,
        T2: GetDataType + Clone + 'a,
    {
        if T::dtype() == T2::dtype() {
            // safety: T and T2 are the same type
            unsafe { std::mem::transmute::<ArbArray<'a, T>, ArbArray<'a, T2>>(self) }
        } else {
            self.view().cast::<T2>().into()
        }
    }

    #[inline]
    pub fn cast_with<T2>(self, _other: &ArbArray<'a, T2>) -> ArbArray<'a, T2>
    where
        T2: GetDataType,
        T: GetDataType,
    {
        if T::dtype() == T2::dtype() {
            // safety: T and T2 are the same type
            unsafe { std::mem::transmute::<ArbArray<'a, T>, ArbArray<'a, T2>>(self) }
        } else {
            unreachable!()
        }
    }

    #[inline]
    pub fn cast_ref_with<T2>(&self, _other: &ArbArray<'a, T2>) -> &ArbArray<'a, T2>
    where
        T2: GetDataType,
        T: GetDataType,
    {
        if T::dtype() == T2::dtype() {
            // safety: T and T2 are the same type
            unsafe { std::mem::transmute::<&ArbArray<'a, T>, &ArbArray<'a, T2>>(self) }
        } else {
            unreachable!()
        }
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn get_type(&self) -> &'static str {
        match self {
            ArbArray::Owned(_) => "Owned Array",
            ArbArray::ViewMut(_) => "ViewMut Array",
            ArbArray::View(_) => "View Array",
            ArbArray::ViewOnBase(_) => "View Array",
            #[cfg(feature = "arw")]
            ArbArray::ArrowChunk(_) => "ArrowChunk Array",
        }
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn strides(&self) -> &[isize] {
        match_arbarray!(self, a, { a.strides() })
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        match_arbarray!(self, a, { a.as_ptr() })
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        match_arbarray!(self, a, { a.as_mut_ptr() }, (ViewMut, Owned))
    }

    #[inline]
    pub fn no_dim0(self) -> Self
    where
        T: Clone,
        ArrOk<'a>: Cast<Self>,
    {
        if self.ndim() == 0 {
            ArbArray::Owned(
                self.into_owned()
                    .0
                    .slice_move(s!(NewAxis))
                    .wrap()
                    .into_dyn(),
            )
        } else {
            self
        }
    }

    #[inline]
    pub fn slice<'b, I: SliceArg<IxDyn>>(&'b self, info: I) -> ArbArray<'b, T>
    where
        'a: 'b,
    {
        // safety: the view has lifetime 'b, this is safe as self exists
        let view: ArrViewD<'b, T> =
            unsafe { std::mem::transmute(self.view().slice(info).into_dyn()) };
        view.into()
    }

    /// Cast the dtype of the array without copy.
    ///
    /// # Safety
    ///
    /// The size of `T` and `T2` must be the same
    #[inline(always)]
    pub unsafe fn into_dtype<T2>(self) -> ArbArray<'a, T2> {
        std::mem::transmute(self)
    }

    #[inline]
    pub fn into_owned(self) -> ArrD<T>
    where
        T: Clone,
        ArrOk<'a>: Cast<Self>,
    {
        match self {
            ArbArray::View(arr) => arr.to_owned(),
            ArbArray::ViewMut(arr) => arr.to_owned(),
            ArbArray::Owned(arr) => arr,
            ArbArray::ViewOnBase(vb) => vb.view().to_owned(),
            #[cfg(feature = "arw")]
            ArbArray::ArrowChunk(_) => {
                let mut arr = self;
                arr.prepare();
                arr.into_owned()
            }
        }
    }

    pub fn into_owned_inner(self) -> TResult<ArrD<T>> {
        if let ArbArray::Owned(arr) = self {
            Ok(arr)
        } else {
            tbail!("ArbArray is not owned")
            // Err("ArbArray is not owned".into())
        }
    }

    /// Convert to f-continuous if there is no performance loss
    pub fn try_to_owned_f(self) -> ArrD<T>
    where
        T: Clone,
        ArrOk<'a>: Cast<Self>,
    {
        use std::mem::MaybeUninit;
        use utils::vec_uninit;
        match self {
            ArbArray::View(arr) => arr.to_owned_f(),
            ArbArray::ViewMut(arr) => {
                if arr.t().is_standard_layout() {
                    arr.to_owned()
                } else {
                    let mut arr_f =
                        Array::from_shape_vec(arr.shape().f(), vec_uninit(arr.len())).unwrap();
                    arr_f.zip_mut_with(&arr, |out, v| *out = MaybeUninit::new(v.clone()));
                    unsafe { arr_f.assume_init().wrap() }
                }
            }
            ArbArray::Owned(arr) => arr,
            ArbArray::ViewOnBase(vb) => vb.view().to_owned_f(),
            #[cfg(feature = "arw")]
            ArbArray::ArrowChunk(_) => {
                let mut arr = self;
                arr.prepare();
                arr.try_to_owned_f()
            }
        }
    }

    /// create an array view of ArrOk.
    #[inline]
    pub fn view(&self) -> ArrViewD<'_, T> {
        match self {
            ArbArray::View(arr) => arr.view(),
            ArbArray::ViewMut(arr) => arr.view(),
            ArbArray::Owned(arr) => arr.view(),
            ArbArray::ViewOnBase(vb) => vb.view().view(),
            #[cfg(feature = "arw")]
            ArbArray::ArrowChunk(_ac) => {
                unreachable!("Can not view ArrowChunk Array before rechunk")
            }
        }
    }

    pub fn view_mut(&mut self) -> ArrViewMutD<'_, T>
    where
        T: Clone,
    {
        match self {
            ArbArray::ViewMut(arr) => arr.view_mut(),
            ArbArray::Owned(arr) => arr.view_mut(),
            ArbArray::View(arr_view) => {
                let arr = arr_view.to_owned();
                *self = ArbArray::Owned(arr);
                self.view_mut()
            }
            ArbArray::ViewOnBase(vb) => {
                let arr = vb.view().to_owned();
                *self = ArbArray::Owned(arr);
                self.view_mut()
            }
            #[cfg(feature = "arw")]
            ArbArray::ArrowChunk(_ac) => {
                unreachable!("Can not viewmut ArrowChunk Array before rechunk")
            }
        }
    }

    pub fn concat<'b>(self, other: Vec<Self>, axis: Axis) -> ArbArray<'b, T>
    where
        T: Clone,
    {
        let arrs = std::iter::once(self.view().no_dim0().0)
            .chain(other.iter().map(|o| o.view().no_dim0().0))
            .collect::<Vec<_>>();
        ArbArray::Owned(ndarray::concatenate(axis, &arrs).unwrap().wrap())
    }

    pub fn stack<'b>(&self, other: Vec<Self>, axis: Axis) -> ArbArray<'b, T>
    where
        T: Clone,
    {
        let arrs = std::iter::once(self.view().no_dim0().0)
            .chain(other.iter().map(|o| o.view().no_dim0().0))
            .collect::<Vec<_>>();
        ArbArray::Owned(ndarray::stack(axis, &arrs).unwrap().wrap())
    }
}
