use crate::lazy::Exprs;
use crate::TpResult;

pub enum GetOutput<'a, 'b> {
    Expr(&'b Exprs<'a>),
    Exprs(Vec<&'b Exprs<'a>>),
}

impl<'a, 'b> From<&'b Exprs<'a>> for GetOutput<'a, 'b> {
    fn from(expr: &'b Exprs<'a>) -> Self {
        GetOutput::Expr(expr)
    }
}

impl<'a, 'b> From<Vec<&'b Exprs<'a>>> for GetOutput<'a, 'b> {
    fn from(exprs: Vec<&'b Exprs<'a>>) -> Self {
        GetOutput::Exprs(exprs)
    }
}

impl<'a, 'b> GetOutput<'a, 'b> {
    pub fn into_exprs(self) -> Vec<&'b Exprs<'a>> {
        match self {
            GetOutput::Expr(expr) => vec![expr],
            GetOutput::Exprs(exprs) => exprs,
        }
    }

    pub fn into_expr(self) -> TpResult<&'b Exprs<'a>> {
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
    pub fn into_exprs(self) -> Vec<&'b mut Exprs<'a>> {
        match self {
            GetMutOutput::Expr(expr) => vec![expr],
            GetMutOutput::Exprs(exprs) => exprs,
        }
    }

    pub fn into_expr(self) -> TpResult<&'b mut Exprs<'a>> {
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
    Expr(&'b mut Exprs<'a>),
    Exprs(Vec<&'b mut Exprs<'a>>),
}

impl<'a, 'b> From<&'b mut Exprs<'a>> for GetMutOutput<'a, 'b> {
    fn from(expr: &'b mut Exprs<'a>) -> Self {
        GetMutOutput::Expr(expr)
    }
}

impl<'a, 'b> From<Vec<&'b mut Exprs<'a>>> for GetMutOutput<'a, 'b> {
    fn from(exprs: Vec<&'b mut Exprs<'a>>) -> Self {
        GetMutOutput::Exprs(exprs)
    }
}

pub enum SetInput<'a> {
    Expr(Exprs<'a>),
    Exprs(Vec<Exprs<'a>>),
}

impl<'a> From<Exprs<'a>> for SetInput<'a> {
    fn from(expr: Exprs<'a>) -> Self {
        SetInput::Expr(expr)
    }
}

impl<'a> From<Vec<Exprs<'a>>> for SetInput<'a> {
    fn from(exprs: Vec<Exprs<'a>>) -> Self {
        SetInput::Exprs(exprs)
    }
}

impl<'a> SetInput<'a> {
    pub fn into_expr(self) -> TpResult<Exprs<'a>> {
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

    pub fn into_exprs(self) -> Vec<Exprs<'a>> {
        match self {
            SetInput::Expr(expr) => vec![expr],
            SetInput::Exprs(exprs) => exprs,
        }
    }
}
