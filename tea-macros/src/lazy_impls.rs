use proc_macro2::TokenStream as TokenStream2;
use quote::quote;

use super::tools::parse_params;

pub(crate) fn impl_view(
    lazy_sig: &syn::Signature,
    arr_type: Option<TokenStream2>,
    _other_type: Option<TokenStream2>,
) -> TokenStream2 {
    let fn_name = &lazy_sig.ident;
    let params = parse_params(lazy_sig);
    let arr_type = if let Some(arr_type) = arr_type {
        arr_type
    } else {
        quote! {}
    };
    quote! {
        #lazy_sig
        {
            self.chain_f_ctx(move |(data, ctx)| {
                let arr = data.view_arr(ctx.as_ref())?;
                match_arrok!(#arr_type arr, a, { Ok((a.view().#fn_name(#(#params),*).into(), ctx)) })
            });
            self
        }
    }
}

pub(crate) fn impl_view2(
    lazy_sig: &syn::Signature,
    arr_type: Option<TokenStream2>,
    other_type: Option<TokenStream2>,
) -> TokenStream2 {
    let fn_name = &lazy_sig.ident;
    let mut params = parse_params(lazy_sig);
    let other = params.remove(0);
    let arr_type = if let Some(arr_type) = arr_type {
        arr_type
    } else {
        quote! {}
    };
    let other_type = if let Some(other_type) = other_type {
        other_type
    } else {
        quote! {}
    };
    quote! {
        #lazy_sig
        {
            self.chain_f_ctx(move |(data, ctx)| {
                let arr = data.view_arr(ctx.as_ref())?;
                let other_arr = #other.view_arr(ctx.as_ref())?;
                match_arrok!(#arr_type arr, a, {
                    match_arrok!(#other_type other_arr, b, {
                        Ok((a.view().#fn_name(&b.view(), #(#params),*).into(), ctx))
                    })
                })
            });
            self
        }
    }
}

pub(crate) fn impl_f64func(
    lazy_sig: &syn::Signature,
    arr_type: Option<TokenStream2>,
    _other_type: Option<TokenStream2>,
) -> TokenStream2 {
    let fn_name = &lazy_sig.ident;
    let params = parse_params(lazy_sig);
    let arr_type = if let Some(arr_type) = arr_type {
        arr_type
    } else {
        quote! {}
    };
    quote! {
        #lazy_sig
        {
            self.chain_f_ctx(move |(data, ctx)| {
                let arr = data.view_arr(ctx.as_ref())?;
                match_arrok!(#arr_type arr, a, { Ok((a.view().map(|v| v.f64().#fn_name(#(#params),*)).into(), ctx)) })
            });
            self
        }
    }
}

pub(crate) fn impl_viewmut(
    lazy_sig: &syn::Signature,
    arr_type: Option<TokenStream2>,
    _other_type: Option<TokenStream2>,
) -> TokenStream2 {
    let fn_name = &lazy_sig.ident;
    let params = parse_params(lazy_sig);
    let arr_type = if let Some(arr_type) = arr_type {
        arr_type
    } else {
        quote! {}
    };
    quote! {
        #lazy_sig
        {
            self.chain_f_ctx(move |(data, ctx)| {
                let mut arr = data.into_arr(ctx.clone())?;
                match_arrok!(#arr_type &mut arr, a, { a.view_mut().#fn_name(#(#params),*) });
                Ok((arr.into(), ctx))
            });
            self
        }
    }
}

pub(crate) fn impl_rolling_by_startidx_agg(
    lazy_sig: &syn::Signature,
    arr_type: Option<TokenStream2>,
    _other_type: Option<TokenStream2>,
) -> TokenStream2 {
    let fn_name = lazy_sig.ident.to_string();
    let fn_str = fn_name
        .strip_prefix("rolling_select_")
        .unwrap_or(fn_name.as_str());
    let fn_name = format!("{}_1d", fn_str);
    let fn_name: syn::Ident = syn::parse_str(&fn_name).unwrap();
    let mut params = parse_params(lazy_sig);
    let roll_start = params.remove(0);
    let arr_type = if let Some(arr_type) = arr_type {
        arr_type
    } else {
        quote! {}
    };
    quote! {
        #lazy_sig
        {
            self.chain_f_ctx(
                move |(data, ctx)| {
                    let arr = data.view_arr(ctx.as_ref())?.deref();
                    let roll_start = #roll_start.view_arr(ctx.as_ref())?.deref().cast_usize();
                    let roll_start_arr = roll_start.view().to_dim1()?;
                    let len = arr.len();
                    if len != roll_start_arr.len() {
                        return Err(format!(
                            "rolling_select_agg: arr.len() != roll_start.len(): {} != {}",
                            arr.len(),
                            roll_start_arr.len()
                        )
                        .into());
                    }

                    let out: ArrOk<'a> = match_arrok!(#arr_type arr, arr, {
                        let arr = arr.view().to_dim1()?;
                        let out = zip(roll_start_arr, 0..len)
                        .map(|(mut start, end)| {
                            if start > end {
                                start = end;  // the start idx should be inbound
                            }
                            let current_arr = arr.slice(s![start..end + 1]).wrap();
                            current_arr.#fn_name(#(#params),*)
                        })
                        .collect_trusted();
                        Arr1::from_vec(out).to_dimd().into()
                    });
                    Ok((out.into(), ctx.clone()))
                }
            );
            self
        }
    }
}

pub(crate) fn impl_rolling_by_startidx_agg2(
    lazy_sig: &syn::Signature,
    arr_type: Option<TokenStream2>,
    other_type: Option<TokenStream2>,
) -> TokenStream2 {
    let fn_name = lazy_sig.ident.to_string();
    let fn_str = fn_name
        .strip_prefix("rolling_select_")
        .unwrap_or(fn_name.as_str());
    let fn_name = format!("{}_1d", fn_str);
    let fn_name: syn::Ident = syn::parse_str(&fn_name).unwrap();
    let mut params = parse_params(lazy_sig);
    let other = params.remove(0);
    let roll_start = params.remove(0);
    let arr_type = if let Some(arr_type) = arr_type {
        arr_type
    } else {
        quote! {}
    };
    let other_type = if let Some(other_type) = other_type {
        other_type
    } else {
        quote! {}
    };
    quote! {
        #lazy_sig
        {
            self.chain_f_ctx(
                move |(data, ctx)| {
                    let arr = data.view_arr(ctx.as_ref())?.deref();
                    let other = #other.view_arr(ctx.as_ref())?.deref();
                    let roll_start = #roll_start.view_arr(ctx.as_ref())?.deref().cast_usize();
                    let roll_start_arr = roll_start.view().to_dim1()?;
                    let len = arr.len();
                    if len != roll_start_arr.len() {
                        return Err(format!(
                            "rolling_select_agg: arr.len() != roll_start.len(): {} != {}",
                            arr.len(),
                            roll_start_arr.len()
                        )
                        .into());
                    }

                    let out: ArrOk<'a> = match_arrok!(#arr_type arr, arr, {
                        let arr = arr.view().to_dim1()?;
                        let out = zip(roll_start_arr, 0..len)
                        .map(|(mut start, end)| {
                            if start > end {
                                start = end;  // the start idx should be inbound
                            }
                            let current_arr = arr.slice(s![start..end + 1]).wrap();
                            match_arrok!(#other_type &other, other, {
                                let other = other.view().to_dim1().unwrap();
                                let other_arr = other.slice(s![start..end + 1]).wrap();
                                current_arr.#fn_name(&other_arr, #(#params),*)
                            })
                        })
                        .collect_trusted();
                        Arr1::from_vec(out).to_dimd().into()
                    });
                    Ok((out.into(), ctx.clone()))
                }
            );
            self
        }
    }
}

pub(crate) fn impl_rolling_by_vecusize_agg(
    lazy_sig: &syn::Signature,
    arr_type: Option<TokenStream2>,
    _other_type: Option<TokenStream2>,
) -> TokenStream2 {
    let fn_name = lazy_sig.ident.to_string();
    let fn_str = fn_name
        .strip_prefix("rolling_select_by_vecusize_")
        .unwrap_or(fn_name.as_str());
    let fn_name = format!("{}_1d", fn_str);
    let fn_name: syn::Ident = syn::parse_str(&fn_name).unwrap();
    let mut params = parse_params(lazy_sig);
    let idxs = params.remove(0);
    let arr_type = if let Some(arr_type) = arr_type {
        arr_type
    } else {
        quote! {}
    };
    quote! {
        #lazy_sig
        {
            self.chain_f_ctx(
                move |(data, ctx)| {
                    let arr = data.view_arr(ctx.as_ref())?.deref();
                    let idxs = #idxs.view_arr(ctx.as_ref())?.deref().cast_vecusize();
                    let idxs_arr = idxs.view().to_dim1()?;
                    let out: ArrOk<'a> = match_arrok!(#arr_type arr, arr, {
                        let arr = arr.view().to_dim1()?;
                        let out: Vec<_> = idxs_arr
                            .iter()
                            .map(|idx| {
                                let current_arr = arr.select_unchecked(Axis(0), idx);
                                current_arr.#fn_name(#(#params),*)
                            })
                            .collect_trusted();
                        Arr1::from_vec(out).to_dimd().into()
                    });
                    Ok((out.into(), ctx.clone()))
                }
            );
            self
        }
    }
}

pub(crate) fn impl_group_by_startidx_agg(
    lazy_sig: &syn::Signature,
    arr_type: Option<TokenStream2>,
    _other_type: Option<TokenStream2>,
) -> TokenStream2 {
    let fn_name = lazy_sig.ident.to_string();
    let fn_str = fn_name
        .strip_prefix("group_by_startidx_")
        .unwrap_or(fn_name.as_str());
    let fn_name = format!("{}_1d", fn_str);
    let fn_name: syn::Ident = syn::parse_str(&fn_name).unwrap();
    let mut params = parse_params(lazy_sig);
    let start_idx = params.remove(0);
    let arr_type = if let Some(arr_type) = arr_type {
        arr_type
    } else {
        quote! {}
    };
    quote! {
        #lazy_sig
        {
            self.chain_f_ctx(
                move |(data, ctx)| {
                    let arr = data.view_arr(ctx.as_ref())?.deref();
                    let group_start = if let Ok(mut group_idx) = #start_idx.view_arr_vec(ctx.as_ref()) {
                        // idx with time info or other info
                        group_idx.pop().unwrap().deref().cast_usize()
                    } else {
                        // only idx info
                        group_idx.view_arr(ctx.as_ref())?.deref().cast_usize()
                    };
                    let group_start_view = group_start.view().to_dim1()?;
                    let out: ArrOk<'a> = match_arrok!(#arr_type arr, arr, {
                        let arr = arr.view().to_dim1()?;
                        let out = group_start_view.as_slice().unwrap().windows(2)
                        .map(|v| {
                            // v: (start, next_start)
                            let current_arr = arr.slice(s![v[0]..v[1]]).wrap();
                            current_arr.#fn_name(#(#params),*)
                        })
                        .collect_trusted();
                        Arr1::from_vec(out).to_dimd().into()
                    });
                    Ok((out.into(), ctx.clone()))
                }
            );
            self
        }
    }
}

pub(crate) fn impl_group_by_startidx_agg2(
    lazy_sig: &syn::Signature,
    arr_type: Option<TokenStream2>,
    other_type: Option<TokenStream2>,
) -> TokenStream2 {
    let fn_name = lazy_sig.ident.to_string();
    let fn_str = fn_name
        .strip_prefix("group_by_startidx_")
        .unwrap_or(fn_name.as_str());
    let fn_name = format!("{}_1d", fn_str);
    let fn_name: syn::Ident = syn::parse_str(&fn_name).unwrap();
    let mut params = parse_params(lazy_sig);
    let other = params.remove(0);
    let start_idx = params.remove(0);
    let arr_type = if let Some(arr_type) = arr_type {
        arr_type
    } else {
        quote! {}
    };
    let other_type = if let Some(other_type) = other_type {
        other_type
    } else {
        quote! {}
    };
    quote! {
        #lazy_sig
        {
            self.chain_f_ctx(
                move |(data, ctx)| {
                    let arr = data.view_arr(ctx.as_ref())?.deref();
                    let other = #other.view_arr(ctx.as_ref())?.deref();
                    let group_start = if let Ok(mut group_idx) = #start_idx.view_arr_vec(ctx.as_ref()) {
                        group_idx.pop().unwrap().deref().cast_usize()
                    } else {
                        group_idx.view_arr(ctx.as_ref())?.deref().cast_usize()
                    };
                    let group_start_view = group_start.view().to_dim1()?;

                    let out: ArrOk<'a> = match_arrok!(#arr_type arr, arr, {
                        let arr = arr.view().to_dim1()?;
                        let out = group_start_view.as_slice().unwrap().windows(2)
                        .map(|v| {
                            let (start, next_start) = (v[0], v[1]);
                            let current_arr = arr.slice(s![start..next_start]).wrap();
                            match_arrok!(#other_type &other, other, {
                                let other = other.view().to_dim1().unwrap();
                                let other_arr = other.slice(s![start..next_start]).wrap();
                                current_arr.#fn_name(&other_arr, #(#params),*)
                            })
                        })
                        .collect_trusted();
                        Arr1::from_vec(out).to_dimd().into()
                    });
                    Ok((out.into(), ctx.clone()))
                }
            );
            self
        }
    }
}
