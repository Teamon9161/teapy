use lazy::Expr;
use statrs::distribution::ContinuousCDF;

#[ext_trait]
impl<'a> ExprStatExt for Expr<'a> {
    fn t_cdf(&mut self, df: Expr<'a>, loc: Option<f64>, scale: Option<f64>) -> &mut Self {
        use statrs::distribution::StudentsT;
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?.cast_f64();
            let df = df
                .view_arr(ctx.as_ref())?
                .deref()
                .cast_f64()
                .into_owned()
                .into_scalar()?;
            let loc = loc.unwrap_or(0.);
            let scale = scale.unwrap_or(1.);
            let n = StudentsT::new(loc, scale, df).unwrap();
            let out = arr.view().map(|v| n.cdf(*v));
            Ok((out.into(), ctx))
        });
        self
    }

    fn norm_cdf(&mut self, mean: Option<f64>, std: Option<f64>) -> &mut Self {
        use statrs::distribution::Normal;
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?.cast_f64();
            let n = Normal::new(mean.unwrap_or(0.), std.unwrap_or(1.)).unwrap();
            let out = arr.view().map(|v| n.cdf(*v));
            Ok((out.into(), ctx))
        });
        self
    }

    fn f_cdf(&mut self, df1: Expr<'a>, df2: Expr<'a>) -> &mut Self {
        use statrs::distribution::FisherSnedecor;
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?.cast_f64();
            let df1 = df1
                .view_arr(ctx.as_ref())?
                .deref()
                .cast_f64()
                .into_owned()
                .into_scalar()?;
            let df2 = df2
                .view_arr(ctx.as_ref())?
                .deref()
                .cast_f64()
                .into_owned()
                .into_scalar()?;
            let n = FisherSnedecor::new(df1, df2).unwrap();
            let out = arr.view().map(|v| n.cdf(*v));
            Ok((out.into(), ctx))
        });
        self
    }
}
