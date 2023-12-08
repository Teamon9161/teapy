use proc_macro2::TokenStream as TokenStream2;
use quote::quote;

#[allow(clippy::vec_box)]
pub(crate) fn reduce_nd_method(
    fn_1d_sig: &syn::Signature,
    params: Vec<Box<syn::Pat>>,
    _ty_1d: Option<Box<syn::Type>>,
) -> TokenStream2 {
    let fn_1d_name = &fn_1d_sig.ident;
    quote! {
        {
            let axis = self.norm_axis(axis);
            if self.is_empty() || self.len_of(axis) == 0 {
                return Arr1::from_vec(vec![]).to_dimd();
            }
            if self.ndim() == 1 {
                return ndarray::arr0(self.view().to_dim1().unwrap().#fn_1d_name(#(#params),*)).wrap().to_dimd();
            }
            if !par {
                Zip::from(self.lanes(axis)).map_collect(move |lane| lane.wrap().#fn_1d_name(#(#params.clone()),*)).wrap().to_dimd()
            } else {
                Zip::from(self.lanes(axis)).par_map_collect(move |lane| lane.wrap().#fn_1d_name(#(#params.clone()),*)).wrap().to_dimd()
            }
        }
    }
}

#[allow(clippy::vec_box)]
pub(crate) fn reduce2_nd_method(
    fn_1d_sig: &syn::Signature,
    mut params: Vec<Box<syn::Pat>>,
    _ty_1d: Option<Box<syn::Type>>,
) -> TokenStream2 {
    let fn_1d_name = &fn_1d_sig.ident;
    let other = params[0].clone();
    params.drain(0..1); // remove rhs parameter of method 1d
    quote! {
        {
            let (lhs, rhs) = if self.ndim() == #other.ndim() && self.shape() == #other.shape() {
                let lhs = self.view().to_dim::<<D as DimMax<D2>>::Output>().unwrap();
                let rhs = #other.view().to_dim::<<D as DimMax<D2>>::Output>().unwrap();
                (lhs, rhs)
            } else {
                self.broadcast_with(&(#other)).unwrap()
            };
            let axis = lhs.norm_axis(axis);
            if lhs.is_empty() || lhs.len_of(axis) == 0 {
                return Arr1::from_vec(vec![]).to_dimd();
            }
            if lhs.ndim() == 1 {
                let rhs = rhs.view().to_dim1().unwrap();
                return ndarray::arr0(lhs.to_dim1().unwrap().#fn_1d_name(&rhs, #(#params),*)).wrap().to_dimd();
            }
            if !par {
                Zip::from(lhs.lanes(axis)).and(rhs.lanes(axis)).map_collect(|lane1, lane2| lane1.wrap().#fn_1d_name(&lane2.wrap(), #(#params.clone()),*)).wrap().to_dimd()
            } else {
                Zip::from(lhs.lanes(axis)).and(rhs.lanes(axis)).par_map_collect(|lane1, lane2| lane1.wrap().#fn_1d_name(&lane2.wrap(), #(#params.clone()),*)).wrap().to_dimd()
            }
        }
    }
}

#[allow(clippy::vec_box)]
pub(crate) fn inplace_nd_method(
    fn_1d_sig: &syn::Signature,
    params: Vec<Box<syn::Pat>>,
    _ty_1d: Option<Box<syn::Type>>,
) -> TokenStream2 {
    let fn_1d_name = &fn_1d_sig.ident;
    quote! {
        {
            let axis = self.norm_axis(axis);
            let ndim = self.ndim();
            if ndim == 1 {
                return self.as_dim1_mut().#fn_1d_name(#(#params),*)
            }
            if !par {
                Zip::from(self.lanes_mut(axis)).for_each(|lane| lane.wrap().#fn_1d_name(#(#params.clone()),*))
            } else {
                Zip::from(self.lanes_mut(axis)).par_for_each(|lane| lane.wrap().#fn_1d_name(#(#params.clone()),*))
            }
        }
    }
}

#[allow(clippy::vec_box)]
pub(crate) fn map_nd_method(
    fn_1d_sig: &syn::Signature,
    mut params: Vec<Box<syn::Pat>>,
    ty_1d: Option<Box<syn::Type>>,
) -> TokenStream2 {
    let ty = ty_1d.unwrap();
    let fn_1d_name = &fn_1d_sig.ident;
    params.drain(0..1); // remove out parameter of method 1d
    quote! {
        {
            let axis = self.norm_axis(axis);
            let f_flag = !self.is_standard_layout();
            let shape = self.raw_dim().into_shape().set_f(f_flag);
            let mut out_arr = Arr::<#ty, D>::uninit(shape);
            let mut out_wr = out_arr.view_mut();
            if self.is_empty() || self.len_of(axis) == 0 {
                // we don't need to do anything
            } else if self.ndim() == 1 {
                // fast path for dim1, we don't need to clone params
                let mut out_wr = out_wr.to_dim1().unwrap();
                self.view().to_dim1().unwrap().#fn_1d_name(&mut out_wr, #(#params),*);
            } else {
                self.apply_along_axis(&mut out_wr, axis, par, move |x_1d, mut out_1d| {
                    x_1d.#fn_1d_name(&mut out_1d, #(#params.clone()),*)
                });
            }
            unsafe{out_arr.assume_init()}.to_dimd()
        }
    }
}

#[allow(clippy::vec_box)]
pub(crate) fn map2_nd_method(
    fn_1d_sig: &syn::Signature,
    mut params: Vec<Box<syn::Pat>>,
    ty_1d: Option<Box<syn::Type>>,
) -> TokenStream2 {
    let ty = ty_1d.unwrap();
    let fn_1d_name = &fn_1d_sig.ident;
    let other = params[0].clone();
    params.drain(0..2); // remove rhs and out parameter of method 1d
    quote! {
        {
            let f_flag = !self.is_standard_layout();
            let (lhs, rhs) = if self.ndim() == #other.ndim() && self.shape() == #other.shape() {
                let lhs = self.view().to_dim::<<D as DimMax<D2>>::Output>().unwrap();
                let rhs = #other.view().to_dim::<<D as DimMax<D2>>::Output>().unwrap();
                (lhs, rhs)
            } else {
                self.broadcast_with(&(#other)).unwrap()
            };
            let axis = lhs.norm_axis(axis);
            let shape = lhs.raw_dim().into_shape().set_f(f_flag);
            let mut out = Arr::<#ty, <D as DimMax<D2>>::Output>::uninit(shape);
            let mut out_wr = out.view_mut();
            if lhs.is_empty() || lhs.len_of(axis) == 0 {
                // we don't need to do anything
            } else if lhs.ndim() == 1 {
                // fast path for dim1, we don't need to clone params
                let mut out_wr = out_wr.to_dim1().unwrap();
                let rhs_1d = rhs.view().to_dim1().unwrap();
                lhs.view().to_dim1().unwrap().#fn_1d_name(&rhs_1d, &mut out_wr, #(#params),*);
            } else {
                lhs.apply_along_axis_with(rhs, &mut out_wr, axis, par, |x_1d, y_1d, mut out_1d| {
                    x_1d.#fn_1d_name(&y_1d, &mut out_1d, #(#params.clone()),*)
                });
            }
            unsafe{out.assume_init()}.to_dimd()
        }
    }
}
