use crate::MethodType;
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, parse_quote, Attribute, FnArg, Generics, Ident, ImplItem, ItemImpl,
    ReturnType, Type,
};

pub(crate) type LazyFunc =
    Box<dyn Fn(&syn::Signature, Option<TokenStream2>, Option<TokenStream2>) -> TokenStream2>;

#[derive(Default)]
pub(crate) struct TpAttrParser {
    lazy_func: Option<LazyFunc>,
    lazy_only: bool,
    lazy_exclude: bool,
    arr_type: Option<TokenStream2>,
    other_type: Option<TokenStream2>,
}

// impl Default for TpAttrParser {
//     fn default() -> Self {
//         Self {
//             lazy_func: None,
//             lazy_only: false,
//             lazy_exclude: false,
//             arr_type: None,
//             other_type: None,
//         }
//     }
// }

impl TpAttrParser {
    fn new(attrs: &mut Vec<Attribute>) -> Self {
        for i in 0..attrs.len() {
            let attr = attrs.get(i).unwrap();
            if let Some(res) = Self::new_from_attr(attr) {
                attrs.remove(i);
                return res;
            }
        }
        Default::default()
    }

    fn new_from_attr(attr: &Attribute) -> Option<Self> {
        let mut lazy_func = None;
        let mut lazy_only = false;
        let mut lazy_exclude = false;
        let mut arr_type = None;
        let mut other_type = None;
        if attr.path().is_ident("lazy_exclude") {
            lazy_exclude = true;
            if let Some(res) = Self::parse_nested_meta(attr) {
                // lazy impl is excluded, do not need lazy_func
                arr_type = res.arr_type;
                other_type = res.other_type;
            }
        } else if attr.path().is_ident("lazy_only") {
            lazy_only = true;
            if let Some(res) = Self::parse_nested_meta(attr) {
                lazy_func = res.lazy_func;
                arr_type = res.arr_type;
                other_type = res.other_type;
            }
        } else if attr.path().is_ident("teapy") {
            if let Some(res) = Self::parse_nested_meta(attr) {
                return Some(res);
            }
        } else {
            return None;
        }
        Some(Self {
            lazy_func,
            lazy_only,
            lazy_exclude,
            arr_type,
            other_type,
        })
    }

    fn parse_nested_meta(attr: &Attribute) -> Option<Self> {
        use super::lazy_impls::*;
        let mut lazy_only = false;
        let mut lazy_exclude = false;
        let mut lazy_func: Option<LazyFunc> = None;
        let mut arr_type = None;
        let mut other_type = None;
        let res = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("lazy") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                let s = s.value();
                match s.as_str() {
                    "view" => lazy_func = Some(Box::new(impl_view)),
                    "view2" => lazy_func = Some(Box::new(impl_view2)),
                    "view_mut" => lazy_func = Some(Box::new(impl_viewmut)),
                    "f64_func" => lazy_func = Some(Box::new(impl_f64func)),
                    "rolling_by_startidx" => {
                        lazy_func = Some(Box::new(impl_rolling_by_startidx_agg))
                    }
                    "rolling_by_startidx2" => {
                        lazy_func = Some(Box::new(impl_rolling_by_startidx_agg2))
                    }
                    "rolling_by_vecusize" => {
                        lazy_func = Some(Box::new(impl_rolling_by_vecusize_agg))
                    }
                    "group_by_startidx_agg" => {
                        lazy_func = Some(Box::new(impl_group_by_startidx_agg))
                    }
                    "group_by_startidx_agg2" => {
                        lazy_func = Some(Box::new(impl_group_by_startidx_agg2))
                    }
                    _ => return Err(meta.error("unsupported attribute")),
                }
            } else if meta.path.is_ident("type") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                let s = s.value();
                let s: Ident = syn::parse_str(s.as_str())?;
                arr_type = Some(quote! {#s});
            } else if meta.path.is_ident("type2") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                let s = s.value();
                let s: Ident = syn::parse_str(s.as_str())?;
                other_type = Some(quote! {#s});
            } else if meta.path.is_ident("lazy_only") {
                lazy_only = true;
            } else if meta.path.is_ident("lazy_exclude") {
                lazy_exclude = true
            }
            Ok(())
        });
        if res.is_ok() {
            Some(Self {
                lazy_func,
                lazy_only,
                lazy_exclude,
                arr_type,
                other_type,
            })
        } else {
            None
        }
    }
}

fn parse_input_attr(attr: TokenStream) -> TpAttrParser {
    let attr: TokenStream2 = attr.into();
    let input_attr: Attribute = parse_quote! { #[teapy(#attr)] };
    TpAttrParser::new_from_attr(&input_attr).unwrap_or_default()
}

pub(crate) fn to_lazy_sig(sig: &syn::Signature, double_input_flag: bool) -> syn::Signature {
    let mut sig = sig.clone();
    let mut_self_arg: FnArg = parse_quote! { &mut self };
    if !sig.inputs.is_empty() {
        sig.inputs[0] = mut_self_arg;
    } else {
        sig.inputs.push(mut_self_arg);
    }
    if double_input_flag {
        let other_arg: FnArg = parse_quote! {other: Self};
        if sig.inputs.len() >= 2 {
            sig.inputs[1] = other_arg;
        } else {
            sig.inputs.push(other_arg);
        }
    }
    sig
}

#[allow(clippy::vec_box)]
pub(crate) fn parse_params(sig: &syn::Signature) -> Vec<Box<syn::Pat>> {
    sig.inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(pat_type) = arg {
                Some(pat_type.pat.clone())
            } else {
                None
            }
        })
        .collect()
}

#[allow(clippy::vec_box, dead_code)]
pub(crate) fn remove_params(params: &mut Vec<Box<syn::Pat>>, names: Vec<&str>) {
    params.retain(|param| {
        if let syn::Pat::Ident(pat_ident) = &**param {
            !names.contains(&pat_ident.ident.to_string().as_str())
        } else {
            true
        }
    })
}

fn remove_inline_attr(attrs: &mut Vec<syn::Attribute>) {
    attrs.retain(|attr| {
        if attr.path().is_ident("inline") {
            return false;
        }
        true
    });
}

#[allow(clippy::single_match)]
fn no_mut_arg(sig: &syn::Signature) -> syn::Signature {
    let mut sig = sig.clone();
    sig.inputs.iter_mut().for_each(|arg| match arg {
        FnArg::Typed(pat_type) => {
            if let syn::Pat::Ident(pat_ident) = &mut *pat_type.pat {
                pat_ident.mutability = None;
            }
        }
        _ => {}
    });
    sig
}

pub(crate) fn sig_1d_to_nd(
    fn_1d: syn::ImplItemFn,
    return_type: Option<TokenStream2>,
    method: MethodType,
) -> (
    syn::Signature,
    syn::Block,
    syn::Signature,
    Option<Box<syn::Type>>,
) {
    let mut fn_sig = fn_1d.sig.clone();
    let fn_1d_block = fn_1d.block;
    // add _1d as suffix
    let mut fn_1d_sig = fn_1d.sig;
    fn_1d_sig.ident = format_ident!("{}_1d", fn_1d_sig.ident);

    // change return type of method nd
    let (ty_1d, return_type) = if let Some(return_type) = return_type {
        (None, return_type)
    } else {
        match &fn_1d_sig.output {
            ReturnType::Type(_, ty) => (Some(ty.clone()), quote! { ArrD<#ty> }),
            _ => {
                if let MethodType::Inplace = method {
                    // inplace function does not have return type
                    (None, quote! { () })
                } else {
                    // 1d function doesn't have a return type
                    // this may occur when impl lazy
                    (None, quote! { () })
                    // panic!("1d function must have a return type")
                }
            }
        }
    };

    // remove out parameter of method nd, also remove generic of out in method nd
    // this is because out is need for 1d map method, and we create an uninit output
    // array for method nd, so we don't need out parameter in method nd
    match method {
        MethodType::Map | MethodType::Map2 => {
            // remove output type of 1d method
            fn_1d_sig.output = ReturnType::Default;
            // remove SO in generics of method nd, this is a generic of output array only in method 1d
            fn_sig.generics.params = fn_sig
                .generics
                .params
                .into_iter()
                .filter(|param| {
                    if let syn::GenericParam::Type(type_param) = param {
                        if type_param.ident == "SO" {
                            return false;
                        }
                    }
                    true
                })
                .collect();
            // remove SO in where clause of method nd
            if let Some(where_clause) = &mut fn_sig.generics.where_clause {
                where_clause.predicates = where_clause
                    .predicates
                    .iter()
                    .filter(|where_predicate| {
                        if let syn::WherePredicate::Type(pred_type) = where_predicate {
                            if let syn::Type::Path(type_path) = &pred_type.bounded_ty {
                                if type_path.path.is_ident("SO") {
                                    return false;
                                }
                            }
                        }
                        true
                    })
                    .cloned()
                    .collect();
            }

            let remove_idx = if let MethodType::Map = method { 1 } else { 2 };
            fn_sig.inputs = fn_sig
                .inputs
                .into_iter()
                .enumerate()
                .filter(|(i, _)| *i != remove_idx)
                .map(|(_, arg)| arg)
                .collect();
        }
        _ => {}
    }
    // add arguments for method nd
    fn_sig.inputs.push(parse_quote! { axis: i32 });
    fn_sig.inputs.push(parse_quote! { par: bool });
    fn_sig.output = parse_quote! { -> #return_type };
    (fn_1d_sig, fn_1d_block, fn_sig, ty_1d)
}

#[allow(clippy::manual_map)]
pub(crate) fn ext_tool(attr: TokenStream, input: TokenStream) -> TokenStream {
    let parse_res = parse_input_attr(attr);
    let item_impl = parse_macro_input!(input as ItemImpl);
    let trait_name = if let Some((_, path, _)) = &item_impl.trait_ {
        path.segments.last().expect("No trait name").ident.clone()
    } else {
        panic!("Expected a trait path");
    };

    let self_type = item_impl.self_ty.clone();
    let generics = item_impl.generics.clone();
    let (lazy_trait_methods, lazy_impls, trait_methods, impl_methods): (
        Vec<_>,
        Vec<_>,
        Vec<_>,
        Vec<_>,
    ) = item_impl
        .items
        .iter()
        .filter_map(|item| {
            if let ImplItem::Fn(method) = item {
                let mut method_attrs = method.attrs.clone();
                let current_parse_res = TpAttrParser::new(&mut method_attrs);
                let current_arr_type = if let Some(arr_type) = current_parse_res.arr_type {
                    Some(arr_type)
                } else {
                    parse_res.arr_type.clone()
                };
                let current_other_type = if let Some(other_type) = current_parse_res.other_type {
                    Some(other_type)
                } else {
                    parse_res.other_type.clone()
                };

                let mut attrs_notinline = method_attrs.clone();
                remove_inline_attr(&mut attrs_notinline);
                let method_sig = &method.sig;
                let method_block = &method.block;
                let trait_method_sig = no_mut_arg(method_sig);
                let trait_method = quote! { #(#attrs_notinline)* #trait_method_sig; };
                let impl_method = quote! { #(#method_attrs)* #method_sig #method_block };
                // impl lazy
                let mut lazy_trait_fn_sig =
                    to_lazy_sig(&trait_method_sig, current_other_type.is_some());
                lazy_trait_fn_sig.output = parse_quote! { -> &mut Self };
                lazy_trait_fn_sig.generics = Default::default();

                let lazy_impl = if let Some(lazy_func) = current_parse_res.lazy_func.as_ref() {
                    Some(lazy_func(
                        &lazy_trait_fn_sig,
                        current_arr_type,
                        current_other_type,
                    ))
                } else if let Some(lazy_func) = parse_res.lazy_func.as_ref() {
                    Some(lazy_func(
                        &lazy_trait_fn_sig,
                        current_arr_type,
                        current_other_type,
                    ))
                } else {
                    None
                };
                let lazy_impl = lazy_impl.map(|lazy_impl| quote! {#(#method_attrs)* #lazy_impl});
                let lazy_trait_method = quote! { #(#attrs_notinline)* #lazy_trait_fn_sig; };
                let lazy_only = parse_res.lazy_only | current_parse_res.lazy_only;
                let lazy_exclude = parse_res.lazy_exclude | current_parse_res.lazy_exclude;
                if lazy_only {
                    Some((lazy_trait_method, lazy_impl, None, None))
                } else if lazy_exclude {
                    Some((
                        lazy_trait_method,
                        None,
                        Some(trait_method),
                        Some(impl_method),
                    ))
                } else {
                    Some((
                        lazy_trait_method,
                        lazy_impl,
                        Some(trait_method),
                        Some(impl_method),
                    ))
                }
            } else {
                None
            }
        })
        .fold(
            (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
            |mut acc, (lazy_trait_method, lazy_impl, trait_method, impl_method)| {
                acc.0.push(lazy_trait_method);
                acc.1.push(lazy_impl);
                acc.2.push(trait_method);
                acc.3.push(impl_method);
                acc
            },
        );
    let (lazy_trait_methods, lazy_impls): (Vec<_>, Vec<_>) = lazy_trait_methods
        .into_iter()
        .zip(lazy_impls)
        .filter_map(|(trait_method, impl_)| {
            if let Some(impl_) = impl_ {
                Some((trait_method, impl_))
            } else {
                None
            }
        })
        .unzip();
    let trait_methods: Vec<TokenStream2> = trait_methods.into_iter().flatten().collect();
    let impl_methods = impl_methods.into_iter().flatten().collect();
    expand_trait_ext(
        trait_name,
        self_type,
        trait_methods,
        impl_methods,
        generics,
        // attr,
        lazy_trait_methods,
        lazy_impls,
    )
}

#[allow(clippy::manual_map)]
pub(crate) fn arr_ext_tool<F>(
    attr: TokenStream,
    input: TokenStream,
    nd_impl_func: F,
    otype: Option<proc_macro2::TokenStream>,
    method: MethodType,
) -> TokenStream
where
    F: Fn(&syn::Signature, Vec<Box<syn::Pat>>, Option<Box<Type>>) -> proc_macro2::TokenStream,
{
    let item_impl = parse_macro_input!(input as ItemImpl);
    let trait_name = if let Some((_, path, _)) = &item_impl.trait_ {
        path.segments.last().expect("No trait name").ident.clone()
    } else {
        panic!("Expected a trait path");
    };
    let parse_res = parse_input_attr(attr);
    let self_type = item_impl.self_ty.clone();
    let generics = item_impl.generics.clone();
    let (lazy_trait_methods, lazy_impls, trait_methods, impl_methods): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) = item_impl.items.into_iter().filter_map(|item| {
        if let ImplItem::Fn(fn_1d) = item {
            let mut fn_1d_attrs = fn_1d.attrs.clone();
            let current_parse_res = TpAttrParser::new(&mut fn_1d_attrs);
            let current_arr_type = if let Some(arr_type) = current_parse_res.arr_type {
                Some(arr_type)
            } else {
                parse_res.arr_type.clone()
            };
            let current_other_type = if let Some(other_type) = current_parse_res.other_type {
                Some(other_type)
            } else {
                parse_res.other_type.clone()
            };
            let mut fn_attrs_notinline = fn_1d_attrs.clone();
            remove_inline_attr(&mut fn_attrs_notinline);
            let (fn_1d_sig, fn_1d_block, fn_sig, ty_1d) = sig_1d_to_nd(fn_1d, otype.clone(), method);
            let params = parse_params(&fn_1d_sig);
            let fn_block = nd_impl_func(&fn_1d_sig, params, ty_1d);
            let trait_fn_sig = no_mut_arg(&fn_sig);
            let trait_fn_1d_sig = no_mut_arg(&fn_1d_sig);
            let trait_method = quote! { #(#fn_attrs_notinline)* #trait_fn_1d_sig; #(#fn_attrs_notinline)* #trait_fn_sig;};
            let impl_method = quote! { #(#fn_1d_attrs)* #fn_1d_sig #fn_1d_block #(#fn_attrs_notinline)* #fn_sig #fn_block};
            let lazy_output: ReturnType = parse_quote!{ -> &mut Self };
            let mut lazy_trait_fn_sig = to_lazy_sig(&fn_sig, current_other_type.is_some());
            lazy_trait_fn_sig.output = lazy_output;
            lazy_trait_fn_sig.generics = Default::default();
            let lazy_impl = if let Some(lazy_func) = current_parse_res.lazy_func.as_ref() {
                Some(lazy_func(&lazy_trait_fn_sig, current_arr_type, current_other_type))
            } else if let Some(lazy_func) = parse_res.lazy_func.as_ref() {
                Some(lazy_func(&lazy_trait_fn_sig, current_arr_type, current_other_type))
            } else {
                None
            };
            let lazy_impl = lazy_impl.map(|lazy_impl| quote!{#(#fn_1d_attrs)* #lazy_impl});
            let lazy_trait_method = quote! { #(#fn_attrs_notinline)* #lazy_trait_fn_sig; };
            let lazy_only = parse_res.lazy_only | current_parse_res.lazy_only;
            let lazy_exclude = parse_res.lazy_exclude | current_parse_res.lazy_exclude;
            if lazy_only {
                Some((lazy_trait_method, lazy_impl, None, None))
            } else if lazy_exclude {
                Some((lazy_trait_method, None, Some(trait_method), Some(impl_method)))
            } else {
                Some((lazy_trait_method, lazy_impl, Some(trait_method), Some(impl_method)))
            }
        } else {
            None
        }
    }).fold(
        (Vec::new(), Vec::new(), Vec::new(), Vec::new()), |mut acc, (lazy_trait_method, lazy_impl, trait_method, impl_method)| {
            acc.0.push(lazy_trait_method);
            acc.1.push(lazy_impl);
            acc.2.push(trait_method);
            acc.3.push(impl_method);
            acc
        }
    );

    // use lazy_impls to determine whether there should be a lazy method
    let (lazy_trait_methods, lazy_impls): (Vec<_>, Vec<_>) = lazy_trait_methods
        .into_iter()
        .zip(lazy_impls)
        .filter_map(|(trait_method, impl_)| {
            if let Some(impl_) = impl_ {
                Some((trait_method, impl_))
            } else {
                None
            }
        })
        .unzip();
    let trait_methods = trait_methods.into_iter().flatten().collect();
    let impl_methods = impl_methods.into_iter().flatten().collect();
    expand_trait_ext(
        trait_name,
        self_type,
        trait_methods,
        impl_methods,
        generics,
        // attr,
        lazy_trait_methods,
        lazy_impls,
    )
}

pub(crate) fn expand_trait_ext(
    trait_name: Ident,
    self_type: Box<Type>,
    trait_methods: Vec<TokenStream2>,
    impl_methods: Vec<TokenStream2>,
    generics: Generics,
    // _attr: TokenStream2,
    lazy_trait_methods: Vec<TokenStream2>,
    lazy_impls: Vec<TokenStream2>,
) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let lazy_trait_name = format_ident!("AutoExpr{}", trait_name);
    let expanded = if (!lazy_impls.is_empty()) && (!trait_methods.is_empty()) {
        quote! {
            // #attr
            pub trait #trait_name #impl_generics {
                #(#trait_methods)*
            }

            impl #impl_generics #trait_name #ty_generics for #self_type #where_clause {
                #(#impl_methods)*
            }

            #[cfg(feature="lazy")]
            pub trait #lazy_trait_name {
                #(#lazy_trait_methods)*
            }
            #[cfg(feature="lazy")]
            impl<'a> #lazy_trait_name for Expr<'a> {
                #(#lazy_impls)*
            }
        }
    } else if lazy_impls.is_empty() {
        quote! {
            // #attr
            pub trait #trait_name #impl_generics {
                #(#trait_methods)*
            }

            impl #impl_generics #trait_name #ty_generics for #self_type #where_clause {
                #(#impl_methods)*
            }
        }
    } else if trait_methods.is_empty() {
        quote! {
            #[cfg(feature="lazy")]
            pub trait #lazy_trait_name {
                #(#lazy_trait_methods)*
            }
            #[cfg(feature="lazy")]
            impl<'a> #lazy_trait_name for Expr<'a> {
                #(#lazy_impls)*
            }
        }
    } else {
        quote! {}
    };
    TokenStream::from(expanded)
}
