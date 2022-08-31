use std::ops::Div;
impl f64 {
    fn div_usize(self, rhs: usize) -> f64 {
        self / rhs as f64
    }
}
fn main(){
    let window = 1_usize;
    let a = 5_f64.div_usize(window);
    println!("{}", a);
}