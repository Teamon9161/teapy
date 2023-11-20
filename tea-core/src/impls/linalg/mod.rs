mod generate;
mod layout;
mod least_squares;
mod svd;
mod utils;

pub use self::utils::replicate;
pub use generate::conjugate;
pub use layout::{into_matrix, transpose, transpose_over, MatrixLayout};
pub use least_squares::LeastSquaresResult;
