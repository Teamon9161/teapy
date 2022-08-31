// use std::path::Path;



fn main() {
    // let library_path = Path::new(r"C:/mingw64/lib/gcc/x86_64-w64-mingw32/12.1.0/include/c++");
    // cc::Build::new()
    //     .file("src/ts_func.cpp")
    //     .cpp(true)
    //     .cpp_link_stdlib("stdc++")
    //     .include("./src")
    //     .include(library_path)
    //     .compile("ts_func");
    cxx_build::bridge("src/ffi/mod.rs")
        .file("src/ts_func.cpp")
        .include("src")
        .flag_if_supported("-std=c++11")
        .compile("ts_func")
}