use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, format_ident};
use syn::{
    parse_quote, parse_macro_input,
    Item, ImplItem, ReturnType, Type, Generics, Ident, FnArg,
};
use crate::MethodType;

pub(crate) fn pre_impl_parser(input: Item) -> (Ident, Box<Type>, Vec<TokenStream2>, Vec<TokenStream2>, Generics)
{   
    if let Item::Impl(item_impl) = input {
        let trait_name = if let Some((_, path, _)) = &item_impl.trait_ {
            path.segments.last().expect("No trait name").ident.clone()
        } else {
            panic!("Expected a trait path");
        };

        let self_type = item_impl.self_ty.clone();
        let generics = item_impl.generics.clone();
        let (trait_methods, impl_methods): (Vec<_>, Vec<_>) = item_impl.items.iter().filter_map(|item| {
            if let ImplItem::Fn(method) = item {
                let method_attrs = &method.attrs;
                let mut attrs_notinline = method.attrs.clone();
                remove_inline_attr(&mut attrs_notinline);
                let method_sig = &method.sig;
                let method_block = &method.block;
                let trait_method_sig = no_mut_arg(method_sig);
                let trait_method = quote! { #(#attrs_notinline)* #trait_method_sig; };
                let impl_method = quote! { #(#method_attrs)* #method_sig #method_block };
                Some((trait_method, impl_method))
            } else {
                None
            }
        }).unzip();
        (trait_name, self_type, trait_methods, impl_methods, generics)
    } else {
        panic!("Expected an impl block")
    }
}


pub(crate) fn parse_params(sig: &syn::Signature) -> Vec<Box<syn::Pat>>
{   
    sig.inputs.iter().filter_map(|arg| {
        if let FnArg::Typed(pat_type) = arg {
            Some(pat_type.pat.clone())
        } else {
            None
        }
    }).collect()
}

#[allow(dead_code)]
pub(crate) fn remove_params(params: &mut Vec<Box<syn::Pat>>, names: Vec<&str>)
{
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
            return false
        }
        true
    });
}

fn no_mut_arg(sig: &syn::Signature) -> syn::Signature {
    let mut sig = sig.clone();
    sig.inputs.iter_mut().for_each(|arg| {
        match arg {
            FnArg::Typed(pat_type) => {
                if let syn::Pat::Ident(pat_ident) = &mut *pat_type.pat {
                    pat_ident.mutability = None;
                }
            }
            _ => {},
        }
    });
    sig
}


pub(crate) fn sig_1d_to_nd(fn_1d: syn::ImplItemFn, return_type: Option<TokenStream2>, method: MethodType) -> (syn::Signature, syn::Block, syn::Signature, Option<Box<syn::Type>>)
{   
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
                    panic!("1d function must have a return type")
                }
            },
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
            fn_sig.generics.params = fn_sig.generics.params.into_iter().filter(|param| {
                if let syn::GenericParam::Type(type_param) = param {
                    if type_param.ident == "SO" {
                        return false
                    }
                }
                true
            }).collect();
            // remove SO in where clause of method nd
            if let Some(where_clause) = &mut fn_sig.generics.where_clause {
                where_clause.predicates = where_clause.predicates.iter().filter(|where_predicate| {
                    if let syn::WherePredicate::Type(pred_type) = where_predicate {
                        if let syn::Type::Path(type_path) = &pred_type.bounded_ty {
                            if type_path.path.is_ident("SO") {
                                return false;
                            }
                        }
                    }
                    true
                }).cloned().collect();
            }
    
            let remove_idx = if let MethodType::Map = method {
                1
            } else {
                2
            };
            fn_sig.inputs = fn_sig.inputs.into_iter().enumerate().filter(|(i, _)| {
                if *i == remove_idx {false} else {true}
            }).map(|(_, arg)| arg).collect();
        }
        _ => {}
    }
    // add arguments for method nd
    fn_sig.inputs.push(parse_quote! { axis: i32 });
    fn_sig.inputs.push(parse_quote! { par: bool });
    fn_sig.output = parse_quote! { -> #return_type };
    (fn_1d_sig, fn_1d_block, fn_sig, ty_1d)
}


pub(crate) fn expand_trait_ext
(
    trait_name: Ident, 
    self_type: Box<Type>, 
    trait_methods: Vec<TokenStream2>, 
    impl_methods: Vec<TokenStream2>, 
    generics: Generics,
    attr: TokenStream2,
) -> TokenStream
{
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let expanded = quote! {
        #attr
        pub trait #trait_name #impl_generics {
            #(#trait_methods)*
        }

        impl #impl_generics #trait_name #ty_generics for #self_type #where_clause {
            #(#impl_methods)*
        }
    };

    TokenStream::from(expanded)
}

pub(crate) fn arr_ext_tool<F>(
    attr: TokenStream, 
    input: TokenStream, 
    nd_impl_func: F, 
    otype: Option<proc_macro2::TokenStream>,
    method: MethodType,
) -> TokenStream
where F: Fn(&syn::Signature, Vec<Box<syn::Pat>>, Option<Box<Type>>) -> proc_macro2::TokenStream
{   
    let input = parse_macro_input!(input as Item);
    if let Item::Impl(item_impl) = input {
        let trait_name = if let Some((_, path, _)) = &item_impl.trait_ {
            path.segments.last().expect("No trait name").ident.clone()
        } else {
            panic!("Expected a trait path");
        };

        let self_type = item_impl.self_ty.clone();
        let generics = item_impl.generics.clone();
        let (trait_methods, impl_methods): (Vec<_>, Vec<_>) = item_impl.items.into_iter().filter_map(|item| {
            if let ImplItem::Fn(fn_1d) = item {
                let fn_1d_attrs = fn_1d.attrs.clone();
                let mut fn_attrs_notinline = fn_1d.attrs.clone();
                remove_inline_attr(&mut fn_attrs_notinline);
                let (fn_1d_sig, fn_1d_block, fn_sig, ty_1d) = sig_1d_to_nd(fn_1d, otype.clone(), method);
                let params = parse_params(&fn_1d_sig);
                let fn_block = nd_impl_func(&fn_1d_sig, params, ty_1d);
                let trait_fn_sig = no_mut_arg(&fn_sig);
                let trait_fn_1d_sig = no_mut_arg(&fn_1d_sig);
                let trait_method = quote! { #(#fn_attrs_notinline)* #trait_fn_1d_sig; #(#fn_attrs_notinline)* #trait_fn_sig;};
                let impl_method = quote! { #(#fn_1d_attrs)* #fn_1d_sig #fn_1d_block #(#fn_attrs_notinline)* #fn_sig #fn_block};
                Some((trait_method, impl_method))
            } else {
                None
            }
        }).unzip();
        expand_trait_ext(trait_name, self_type, trait_methods, impl_methods, generics, attr.into())
    } else {
        panic!("Expected an impl block")
    }
}

