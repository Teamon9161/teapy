#define DBL_MAX 1.7976931348623158e+308
#define DBL_MIN 2.2250738585072014e-308
#define ISNAN(x) ((x) != (x))
#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))

void ts_sma_1d(const double *arr, double *orr, int length, int window, int min_periods, int ostep);
void ts_sum_1d(const double *arr, double *orr, int length, int window, int min_periods, int ostep);
void ts_prod_mean_1d(const double *arr, double *orr, int length, int window, int min_periods, int ostep);
void ts_prod_1d(const double *arr, double *orr, int length, int window, int min_periods, int ostep);
void ts_ema_1d(const double *arr, double *orr,const double *weight, int length, int window, int min_periods, int ostep);
void ts_max_1d(const double *arr, double *orr, int length, int window, int min_periods, int ostep);
void ts_argmax_1d(const double *arr, double *orr, int length, int window, int min_periods, int ostep);
void ts_min_1d(const double *arr, double *orr, int length, int window, int min_periods, int ostep);
void ts_argmin_1d(const double *arr, double *orr, int length, int window, int min_periods, int ostep);
void ts_std_1d(const double *arr, double *orr, int length, int window, int min_periods, int ostep);
void ts_skew_1d(const double *arr, double *orr, int length, int window, int min_periods, int ostep);
void ts_kurt_1d(const double *arr, double *orr, int length, int window, int min_periods, int ostep);
void ts_rank_1d(const double *arr, double *orr, int length, int window, int min_periods, int ostep);
void ts_stable_1d(const double *arr, double *orr, int length, int window, int min_periods, int ostep);
void ts_minmaxnorm_1d(const double *arr, double *orr, int length, int window, int min_periods, int ostep);
void ts_meanstdnorm_1d(const double *arr, double *orr, int length, int window, int min_periods, int ostep);
void ts_cov_1d(const double *arr, const double *brr, double *orr, int length, int window, int min_periods, int ostep);
void ts_corr_1d(const double *arr, const double *brr, double *orr, int length, int window, int min_periods, int ostep);
void ts_reg_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step);
void ts_tsf_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step);
void ts_reg_slope_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step);
void ts_reg_intercept_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step);
void ts_wma_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step);