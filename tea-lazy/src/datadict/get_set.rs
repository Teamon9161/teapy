use crate::Expr;
use core::error::TpResult;

pub enum GetOutput<'a, 'b> {
    Expr(&'b Expr<'a>),
    Exprs(Vec<&'b Expr<'a>>),
}

impl<'a, 'b> From<&'b Expr<'a>> for GetOutput<'a, 'b> {
    #[inline(always)]
    fn from(expr: &'b Expr<'a>) -> Self {
        GetOutput::Expr(expr)
    }
}

impl<'a, 'b> From<Vec<&'b Expr<'a>>> for GetOutput<'a, 'b> {
    #[inline(always)]
    fn from(exprs: Vec<&'b Expr<'a>>) -> Self {
        GetOutput::Exprs(exprs)
    }
}

impl<'a, 'b> GetOutput<'a, 'b> {
    #[inline]
    pub fn into_exprs(self) -> Vec<&'b Expr<'a>> {
        match self {
            GetOutput::Expr(expr) => vec![expr],
            GetOutput::Exprs(exprs) => exprs,
        }
    }

    #[inline]
    pub fn into_expr(self) -> TpResult<&'b Expr<'a>> {
        match self {
            GetOutput::Expr(expr) => Ok(expr),
            GetOutput::Exprs(mut exprs) => {
                if exprs.len() == 1 {
                    Ok(exprs.pop().unwrap())
                } else {
                    Err("The output should not be a vector of expressions!".into())
                }
            }
        }
    }
}

impl<'a, 'b> GetMutOutput<'a, 'b> {
    #[inline]
    pub fn into_exprs(self) -> Vec<&'b mut Expr<'a>> {
        match self {
            GetMutOutput::Expr(expr) => vec![expr],
            GetMutOutput::Exprs(exprs) => exprs,
        }
    }

    #[inline]
    pub fn into_expr(self) -> TpResult<&'b mut Expr<'a>> {
        match self {
            GetMutOutput::Expr(expr) => Ok(expr),
            GetMutOutput::Exprs(mut exprs) => {
                if exprs.len() == 1 {
                    Ok(exprs.pop().unwrap())
                } else {
                    Err("The output should not be a vector of expressions!".into())
                }
            }
        }
    }
}

pub enum GetMutOutput<'a, 'b> {
    Expr(&'b mut Expr<'a>),
    Exprs(Vec<&'b mut Expr<'a>>),
}

impl<'a, 'b> From<&'b mut Expr<'a>> for GetMutOutput<'a, 'b> {
    #[inline(always)]
    fn from(expr: &'b mut Expr<'a>) -> Self {
        GetMutOutput::Expr(expr)
    }
}

impl<'a, 'b> From<Vec<&'b mut Expr<'a>>> for GetMutOutput<'a, 'b> {
    #[inline(always)]
    fn from(exprs: Vec<&'b mut Expr<'a>>) -> Self {
        GetMutOutput::Exprs(exprs)
    }
}

pub enum SetInput<'a> {
    Expr(Expr<'a>),
    Exprs(Vec<Expr<'a>>),
}

impl<'a> From<Expr<'a>> for SetInput<'a> {
    #[inline(always)]
    fn from(expr: Expr<'a>) -> Self {
        SetInput::Expr(expr)
    }
}

impl<'a> From<Vec<Expr<'a>>> for SetInput<'a> {
    #[inline(always)]
    fn from(exprs: Vec<Expr<'a>>) -> Self {
        SetInput::Exprs(exprs)
    }
}

impl<'a> SetInput<'a> {
    #[inline]
    pub fn into_expr(self) -> TpResult<Expr<'a>> {
        match self {
            SetInput::Expr(expr) => Ok(expr),
            SetInput::Exprs(mut exprs) => {
                if exprs.len() == 1 {
                    Ok(exprs.pop().unwrap())
                } else {
                    Err("The input should not be a vector of expressions!".into())
                }
            }
        }
    }

    #[inline]
    pub fn into_exprs(self) -> Vec<Expr<'a>> {
        match self {
            SetInput::Expr(expr) => vec![expr],
            SetInput::Exprs(exprs) => exprs,
        }
    }
}
