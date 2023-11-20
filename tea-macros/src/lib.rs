extern crate proc_macro;

mod tools;
mod methods_impl;

use tools::{pre_impl_parser, arr_ext_tool, expand_trait_ext};
use proc_macro::TokenStream;
use syn::{parse_macro_input, Item};

#[derive(Clone, Copy)]
pub(crate) enum MethodType {
    Reduce,
    Map,
    Map2,
    Inplace,
}


#[proc_macro_attribute]
pub fn ext_trait(attr: TokenStream, input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as Item);
    let (trait_name, self_type, trait_methods, impl_methods, generics) = pre_impl_parser(input);
    expand_trait_ext(trait_name, self_type, trait_methods, impl_methods, generics, attr.into())
}

#[proc_macro_attribute]
pub fn arr_agg_ext(attr: TokenStream, input: TokenStream) -> TokenStream {
    arr_ext_tool(attr, input, methods_impl::reduce_nd_method, None, MethodType::Reduce)
}

#[proc_macro_attribute]
pub fn arr_agg2_ext(attr: TokenStream, input: TokenStream) -> TokenStream {
    arr_ext_tool(attr, input, methods_impl::reduce2_nd_method, None, MethodType::Reduce)
}

#[proc_macro_attribute]
pub fn arr_map_ext(attr: TokenStream, input: TokenStream) -> TokenStream {
    arr_ext_tool(attr, input, methods_impl::map_nd_method, None, MethodType::Map)
}

#[proc_macro_attribute]
pub fn arr_map2_ext(attr: TokenStream, input: TokenStream) -> TokenStream {
    arr_ext_tool(attr, input, methods_impl::map2_nd_method, None, MethodType::Map2)
}

#[proc_macro_attribute]
pub fn arr_inplace_ext(attr: TokenStream, input: TokenStream) -> TokenStream {
    arr_ext_tool(attr, input, methods_impl::inplace_nd_method, None, MethodType::Inplace)
}



