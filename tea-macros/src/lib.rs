extern crate proc_macro;

mod lazy_impls;
mod methods_impls;
mod tools;

use proc_macro::TokenStream;
// use syn::{parse_macro_input, Item};
use tools::{arr_ext_tool, ext_tool};

#[derive(Clone, Copy)]
pub(crate) enum MethodType {
    Reduce,
    Reduce2,
    Map,
    Map2,
    Inplace,
}

#[proc_macro_attribute]
pub fn ext_trait(attr: TokenStream, input: TokenStream) -> TokenStream {
    ext_tool(attr, input)
}

#[proc_macro_attribute]
pub fn arr_agg_ext(attr: TokenStream, input: TokenStream) -> TokenStream {
    arr_ext_tool(
        attr,
        input,
        methods_impls::reduce_nd_method,
        None,
        MethodType::Reduce,
    )
}

#[proc_macro_attribute]
pub fn arr_agg2_ext(attr: TokenStream, input: TokenStream) -> TokenStream {
    arr_ext_tool(
        attr,
        input,
        methods_impls::reduce2_nd_method,
        None,
        MethodType::Reduce2,
    )
}

#[proc_macro_attribute]
pub fn arr_map_ext(attr: TokenStream, input: TokenStream) -> TokenStream {
    arr_ext_tool(
        attr,
        input,
        methods_impls::map_nd_method,
        None,
        MethodType::Map,
    )
}

#[proc_macro_attribute]
pub fn arr_map2_ext(attr: TokenStream, input: TokenStream) -> TokenStream {
    arr_ext_tool(
        attr,
        input,
        methods_impls::map2_nd_method,
        None,
        MethodType::Map2,
    )
}

#[proc_macro_attribute]
pub fn arr_inplace_ext(attr: TokenStream, input: TokenStream) -> TokenStream {
    arr_ext_tool(
        attr,
        input,
        methods_impls::inplace_nd_method,
        None,
        MethodType::Inplace,
    )
}
