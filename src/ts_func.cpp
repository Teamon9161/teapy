#include <cmath>
#include <teapy/include/ts_func.h>
// enum VALUE{NAN_,ZERO_,ONE_,SELF_};

inline void nan_1d(double *orr, int length, int o_step, double value)
{
    int i = 0;
    while (i < length)
    {
        *orr = value;
        orr += o_step;
        i++;
    }
        
}

inline bool check_window(int window, int min_periods, double *orr, int length, int o_step = 1, double value=NAN)
{
    if (window < min_periods)
    {
        nan_1d(orr, length, o_step, value);
        return true;
    }
    return false;
}

void ts_sma_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step)
{
    double sum = 0, _;
    int i = 0, start = 0, end = window - 1, tmp_window = window;
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    while (i < end)
    {   
        _ = arr[i++];
        ISNAN(_) ? tmp_window -= 1 : sum += _;
        *orr = NAN;
        orr += o_step;
    }

    for (; end < length; start++, end++, orr+=o_step)
    {
        // if (ISNAN(sum)) {sum = 0;}
        _ = arr[end];
        ISNAN(_) ? tmp_window -= 1 : sum += _;
        *orr = (tmp_window >= min_periods) ? (sum / tmp_window) : NAN;
        _ = arr[start];
        ISNAN(_) ? tmp_window += 1 : sum -= _;
    }
}

void ts_sum_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step)
{
    double sum = 0;
    int i = 0, start = 0, end = window - 1, tmp_window = window;
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    while (i < end)
    {
        ISNAN(arr[i]) ? (tmp_window -= 1) : (sum += arr[i]);
        // orr[i++] = NAN;
        *orr = NAN;
        orr += o_step;
        i++;
    }

    for (; end < length; start++, end++, orr+=o_step)
    {
        if (ISNAN(sum))
            sum = 0;
        ISNAN(arr[end]) ? (tmp_window -= 1) : sum += arr[end];
        // orr[end] = (tmp_window >= min_periods) ? (sum) : NAN;
        *orr = (tmp_window >= min_periods) ? (sum) : NAN;
        ISNAN(arr[start]) ? tmp_window += 1 : sum -= arr[start];
    }
}

void ts_prod_mean_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step)
{
    double prod = 1.;
    int i = 0, start = 0, end = window - 1, tmp_window = window;
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    while (i < end)
    {
        ISNAN(arr[i]) ? (tmp_window -= 1) : (prod *= arr[i]);
        *orr = NAN;
        orr += o_step;
        i++;
    }

    for (; end < length; start++, end++, orr+=o_step)
    {
        if (ISNAN(prod))
            prod = 1;
        ISNAN(arr[end]) ? (tmp_window -= 1) : prod *= arr[end];
        *orr = (tmp_window >= min_periods) ? pow(prod, 1. / tmp_window) : NAN;
        ISNAN(arr[start]) ? tmp_window += 1 : prod /= arr[start];
    }
}

void ts_prod_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step)
{
    double prod = 1.;
    int i = 0, start = 0, end = window - 1, tmp_window = window;
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    while (i < end)
    {
        ISNAN(arr[i]) ? (tmp_window -= 1) : (prod *= arr[i]);
        // orr[i++] = NAN;
        *orr = NAN;
        orr += o_step;
        i++;
    }

    for (; end < length; start++, end++, orr+=o_step)
    {
        if (ISNAN(prod))
            prod = 1;
        ISNAN(arr[end]) ? (tmp_window -= 1) : prod *= arr[end];
        // orr[end] = (tmp_window >= min_periods) ? prod : NAN;
        *orr = (tmp_window >= min_periods) ? prod : NAN;
        ISNAN(arr[start]) ? tmp_window += 1 : prod /= arr[start];
    }
}

// 待解决问题：有nan会导致weight之和不为1，为了保持和之前算子的结果一致性先忽略。
void ts_ema_1d(const double *arr, double *orr, const double *weight, int length, int window, int min_periods, int o_step)
{
    double temp, tmp1, weight_sum, _;
    int i = 0, nan_num = 0;
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    while (i < window - 1)
    {
        *orr = NAN;
        orr += o_step;
        i++;
    }
    i = 0;
    while (i < length - window + 1)
    {
        nan_num = 0;
        temp = 0;
        weight_sum = 0;
        for (int j = 0; j < window; j++)
        {   
            tmp1 = arr[i + j];
            if(ISNAN(tmp1))
            {
                nan_num++;
                continue;
            }
            else
            {   
                _ = weight[j];
                temp += tmp1 * _;
                weight_sum += _;
            }
        }
        if (nan_num == window)
            temp = NAN;
        else if(nan_num > 0) 
            temp /= weight_sum;
        *orr = (nan_num + min_periods > window) ? NAN : temp;
        i++;
        orr += o_step;
    }
}

void ts_max_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step)
{
    double temp = DBL_MIN, _;
    int start = 0, end = window - 1, tmp_window = window, temp_idx = -1, i=0;    
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    while (i < end)
    {   
        _ = arr[i];
        tmp_window -= ISNAN(_);
        if(_ > temp){
            temp = _;
            temp_idx = i;
        }
        *orr = NAN;
        orr += o_step;
        i++;
    }

    for (; end < length; start++, end++, orr+=o_step)
    {   
        _ = arr[end];
        tmp_window -= ISNAN(_);
        if (temp_idx < start)
        {   
            _ = arr[start];
            temp = ISNAN(_) ? DBL_MIN : _;
            temp_idx = start;
            i = temp_idx;
            while(++i <= end)
            {
                _ = arr[i];
                if(_ >= temp){
                    temp = _;
                    temp_idx = i;
                }
            }
        }
        else if(_ >= temp)
        {
            temp = _;
            temp_idx = end;
        }
        *orr = (tmp_window >= min_periods) ? temp : NAN;
        tmp_window += ISNAN(arr[start]);
    }
}

// 注意加了1
void ts_argmax_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step)
{
    double temp = DBL_MIN, _;
    int start = 0, end = window - 1, tmp_window = window, temp_idx = -1, i=0;    
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    while (i < end)
    {   
        _ = arr[i];
        tmp_window -= ISNAN(_);
        if(_ > temp){
            temp = _;
            temp_idx = i;
        }
        *orr = NAN;
        orr += o_step;
        i++;
    }

    for (; end < length; start++, end++, orr+=o_step)
    {   
        _ = arr[end];
        tmp_window -= ISNAN(_);
        if (temp_idx < start)
        {   
            _ = arr[start];
            temp = ISNAN(_) ? DBL_MIN : _;
            temp_idx = start;
            i = temp_idx;
            while(++i <= end)
            {
                _ = arr[i];
                if(_ >= temp){
                    temp = _;
                    temp_idx = i;
                }
            }
        }
        else if(_ >= temp)
        {
            temp = _;
            temp_idx = end;
        }
        *orr = (tmp_window >= min_periods) ? temp_idx - start + 1: NAN;
        tmp_window += ISNAN(arr[start]);
    }
}

void ts_min_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step)
{
    double temp = DBL_MAX, _;
    int start = 0, end = window - 1, tmp_window = window, temp_idx = -1, i=0;    
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    while (i < end)
    {   
        _ = arr[i];
        tmp_window -= ISNAN(_);
        if(_ < temp){
            temp = _;
            temp_idx = i;
        }
        *orr = NAN;
        orr += o_step;
        i++;
    }

    for (; end < length; start++, end++, orr+=o_step)
    {   
        _ = arr[end];
        tmp_window -= ISNAN(_);
        if (temp_idx < start)
        {   
            _ = arr[start];
            temp = ISNAN(_) ? DBL_MAX : _;
            temp_idx = start;
            i = temp_idx;
            while(++i <= end)
            {
                _ = arr[i];
                if(_ <= temp){
                    temp = _;
                    temp_idx = i;
                }
            }
        }
        else if(_ <= temp)
        {
            temp = _;
            temp_idx = end;
        }
        *orr = (tmp_window >= min_periods) ? temp : NAN;
        tmp_window += ISNAN(arr[start]);
    }
}

void ts_argmin_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step)
{
    double temp = DBL_MAX, _;
    int start = 0, end = window - 1, tmp_window = window, temp_idx = -1, i=0;    
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    while (i < end)
    {   
        _ = arr[i];
        tmp_window -= ISNAN(_);
        if(_ < temp){
            temp = _;
            temp_idx = i;
        }
        *orr = NAN;
        orr += o_step;
        i++;
    }

    for (; end < length; start++, end++, orr+=o_step)
    {   
        _ = arr[end];
        tmp_window -= ISNAN(_);
        if (temp_idx < start)
        {   
            _ = arr[start];
            temp = ISNAN(_) ? DBL_MAX : _;
            temp_idx = start;
            i = temp_idx;
            while(++i <= end)
            {
                _ = arr[i];
                if(_ <= temp){
                    temp = _;
                    temp_idx = i;
                }
            }
        }
        else if(_ <= temp)
        {
            temp = _;
            temp_idx = end;
        }
        *orr = (tmp_window >= min_periods) ? temp_idx - start + 1 : NAN;
        tmp_window += ISNAN(arr[start]);
    }
}

void ts_std_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step)
{
    double _, sum = 0, sum2 = 0, mean2 = 0;
    int i, start = 0, end = window - 1, tmp_window = window;
    min_periods = max(min_periods, 2);
    if (window == 1){
        check_window(window, 1, orr, length, o_step, 0);
        return;
    }
    else if(check_window(window, min_periods, orr, length, o_step, NAN))
        return;
    for (i = start; i < end; i++, orr+=o_step)
    {
        *orr = NAN;
        if (ISNAN(arr[i]))
            tmp_window -= 1;
        else
        {
            _ = arr[i];
            sum += _;
            sum2 += _ * _;
        }
    }

    for (; end < length; start++, end++, orr+=o_step)
    {
        if (ISNAN(sum))
        {
            sum = 0;
            sum2 = 0;
        }
        if (ISNAN(arr[end]))
            tmp_window -= 1;
        else
        {
            _ = arr[end];
            sum += _;
            sum2 += _ * _;
        }
        if (tmp_window >= min_periods)
        {
            mean2 = sum2 / tmp_window;
            _ = sum / tmp_window; // mean
            _ *= _;
            mean2 -= _;    // variance
        }
        else
            mean2 = NAN;
        *orr = sqrt(mean2 * tmp_window / (tmp_window - 1));
        if (ISNAN(arr[start]))
            tmp_window += 1;
        else
        {
            _ = arr[start];
            sum -= _;
            sum2 -= _ * _;
        }
    }
}

void ts_skew_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step)
{
    double _, sum = 0, sum2 = 0, sum3 = 0, mean, mean2, mean3;
    int i, start = 0, end = window - 1, tmp_window = window;
    min_periods = max(min_periods, 3);
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    for (i = start; i < end; i++, orr+=o_step)
    {
        // orr[i] = NAN;
        *orr = NAN;
        if (ISNAN(arr[i]))
            tmp_window -= 1;
        else
        {
            _ = arr[i];
            sum += _;
            sum2 += _ * _;
            sum3 += _ * _ * _;
        }
    }

    for (; end < length; start++, end++, orr+=o_step)
    {
        if (ISNAN(sum))
        {
            sum = 0;
            sum2 = 0;
            sum3 = 0;
        }
        if (ISNAN(arr[end]))
            tmp_window -= 1;
        else
        {
            _ = arr[end];
            sum += _;
            sum2 += _ * _;
            sum3 += _ * _ * _;
        }
        mean2 = sum2 / tmp_window;
        mean3 = sum3 / tmp_window;
        if (tmp_window >= min_periods)
        {
            mean = sum / tmp_window; // mean
            mean2 -= mean * mean;    // variance
            mean2 = sqrt(mean2);     //std
            _ = mean / mean2;        // mean / std
            mean3 = (mean2 > 0) ? (mean3 / (mean2 * mean2 * mean2) - 3 * _ - _ * _ * _) : NAN;
        }
        else
            mean3 = NAN;
        mean3 = (mean3 < 10e-11 && mean3 > -10e-11) ? 0 : mean3;
        // orr[end] = (tmp_window > 2) ? mean3 * sqrt(tmp_window * (tmp_window - 1)) / (tmp_window - 2) : NAN;
        *orr = (tmp_window > 2) ? mean3 * sqrt(tmp_window * (tmp_window - 1)) / (tmp_window - 2) : NAN;
        if (ISNAN(arr[start]))
            tmp_window += 1;
        else
        {
            _ = arr[start];
            sum -= _;
            sum2 -= _ * _;
            sum3 -= _ * _ * _;
        }
    }
}

void ts_kurt_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step)
{
    double _, _1, sum = 0, sum2 = 0, mean, mean2;
    int i, start = 0, end = window - 1, tmp_window = window;
    min_periods = max(min_periods, 4);
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    for (i = start; i < end; i++, orr+=o_step)
    {
        *orr = NAN;
        if (ISNAN(arr[i]))
            tmp_window -= 1;
        else
        {
            _ = arr[i];
            sum += _;
            sum2 += _ * _;
        }
    }

    for (; end < length; start++, end++, orr+=o_step)
    {
        if (ISNAN(sum))
        {
            sum = 0;
            sum2 = 0;
        }
        if (ISNAN(arr[end]))
            tmp_window -= 1;
        else
        {
            _ = arr[end];
            sum += _;
            sum2 += _ * _;
        }
        if (tmp_window >= min_periods)
        {
            mean2 = sum2 / tmp_window;
            mean = sum / tmp_window; // mean
            mean2 -= mean * mean;    // variance
            _ = 0;
            if (mean2 > 0) // 方差不能为0
            {
                for (int j = start; j <= end; j++)
                {
                    if (!ISNAN(arr[j]))
                    {
                        _1 = arr[j] - mean;
                        _1 *= _1;
                        _ += _1 * _1;
                    }
                }
                _ /= mean2 * mean2;
                
            }
        }
        else
            _ = NAN;
        int n = tmp_window;
        *orr = (n > min_periods) ? 1.0 / (n - 2) / (n - 3) * ((n * n - 1.0) * _ / n - 3 * (n - 1) * (n - 1)) : NAN;
        if (ISNAN(arr[start]))
            tmp_window += 1;
        else
        {
            _ = arr[start];
            sum -= _;
            sum2 -= _ * _;
        }
    }
}

void ts_rank_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step)
{
    double temp, _, _1;
    int i = 0, nan_num = 0, repeat_num = 1;
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    while (i < window - 1)
    {
        // orr[i++] = NAN;
        *orr = NAN;
        orr += o_step;
        i++;
    }
        
    i = 0;
    while (i < length - window + 1)
    {
        nan_num = 0;
        repeat_num = 1;
        temp = 1;                 // 假设其为第一名，每当有元素比他小就加1
        _1 = arr[i + window - 1]; // 记录当前元素大小
        for (int j = 0; j < window - 1; j++)
        {
            _ = arr[i + j];
            if (ISNAN(_))
                nan_num++;
            else if (_ < _1)
                temp++;
            else if (_ == _1)
                repeat_num++;
        }
        // method：pandas中的average
        temp = (nan_num + min_periods > window) ? NAN : temp + 0.5 * (repeat_num - 1);
        // orr[window - 1 + i++] = temp;
        *orr = temp;
        orr += o_step;
        i++;
    }
}

void ts_stable_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step)
{
    double _, sum = 0, sum2 = 0, mean, mean2;
    int i, start = 0, end = window - 1, tmp_window = window;
    min_periods = max(min_periods, 2);
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    for (i = start; i < end; i++, orr+=o_step)
    {
        // orr[i] = NAN;
        *orr = NAN;
        if (ISNAN(arr[i]))
            tmp_window -= 1;
        else
        {
            _ = arr[i];
            sum += _;
            _ *= _;
            sum2 += _;
        }
    }

    for (; end < length; start++, end++, orr+=o_step)
    {
        if (ISNAN(sum))
        {
            sum = 0;
            sum2 = 0;
        }
        if (ISNAN(arr[end]))
            tmp_window -= 1;
        else
        {
            _ = arr[end];
            sum += _;
            sum2 += _ * _;
        }
        if (tmp_window >= min_periods)
        {
            mean2 = sum2 / tmp_window;
            mean = sum / tmp_window; // mean
            mean2 -= mean * mean;    // variance
            _ = (mean2 > 0) ? (mean / sqrt(mean2)) : NAN;
        }
        else
            _ = NAN;
        // orr[end] = _;
        *orr = _;
        if (ISNAN(arr[start]))
            tmp_window += 1;
        else
        {
            _ = arr[start];
            sum -= _;
            sum2 -= _ * _;
        }
    }
}

void ts_minmaxnorm_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step)
{
    double temp = DBL_MIN, temp1 = DBL_MAX, _;
    int start = 0, end = window - 1, tmp_window = window, temp_idx = -1, temp1_idx = -1, i=0;    
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    while (i < end)
    {   
        _ = arr[i]; // 第i个元素
        tmp_window -= ISNAN(_); // 是nan的话窗口期-1
        if(_ > temp){  // 第i个元素大于记录的最大值，更新最大值temp和索引temp_idx
            temp = _;
            temp_idx = i;
        }
        else if (_ < temp1){  // 第i个元素小于记录的最小值，更新最大值temp和索引temp_idx
            temp1 = _;
            temp1_idx = i;
        }
        *orr = NAN;
        orr += o_step;
        i++;
    }

    for (; end < length; start++, end++, orr+=o_step)
    {   
        _ = arr[end];
        tmp_window -= ISNAN(_);
        if (temp_idx < start || temp1_idx < start) // 如果记录的最大值或最小值的索引小于当前期间的开始索引
        {   
            _ = arr[start];
            if (ISNAN(_)){
                temp = DBL_MIN;
                temp1 = DBL_MAX;
            }
            temp_idx = start;
            temp1_idx = start;
            i = temp_idx;
            while(++i <= end) // 重新找到当前区间最大值的位置
            {
                _ = arr[i];
                if(_ >= temp){
                    temp = _;
                    temp_idx = i;
                }
                else if(_ <= temp1){
                    temp1 = _;
                    temp1_idx = i;
                }
            }
        }
        else if(_ >= temp) // 更新最大值
        {
            temp = _;
            temp_idx = end;
        }
        else if(_ <= temp1) // 更新最小值
        {
            temp1 = _;
            temp1_idx = end;
        }
        *orr = (tmp_window >= min_periods) ? (_ - temp1) / (temp - temp1) : NAN;
        tmp_window += ISNAN(arr[start]);
    }
}

void ts_meanstdnorm_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step)
{
    double _, sum = 0, sum2 = 0, mean, mean2;
    int i, start = 0, end = window - 1, tmp_window = window;
    min_periods = max(min_periods, 2);
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    for (i = start; i < end; i++, orr+=o_step)
    {
        // orr[i] = NAN;
        *orr = NAN;
        if (ISNAN(arr[i]))
            tmp_window -= 1;
        else
        {
            _ = arr[i];
            sum += _;
            _ *= _;
            sum2 += _;
        }
    }

    for (; end < length; start++, end++, orr+=o_step)
    {
        if (ISNAN(sum))
        {
            sum = 0;
            sum2 = 0;
        }
        if (ISNAN(arr[end]))
            tmp_window -= 1;
        else
        {
            _ = arr[end];
            sum += _;
            sum2 += _ * _;
        }
        if (tmp_window >= min_periods)
        {
            mean2 = sum2 / tmp_window;
            mean = sum / tmp_window; // mean
            mean2 -= mean * mean;    // variance
            _ = (mean2 > 0) ? ((arr[end] - mean) / sqrt(mean2)) : mean;
        }
        else
            _ = NAN;
        // orr[end] = _;
        *orr = _;
        if (ISNAN(arr[start]))
            tmp_window += 1;
        else
        {
            _ = arr[start];
            sum -= _;
            sum2 -= _ * _;
        }
    }
}

void ts_cov_1d(const double *arr, const double *brr, double *orr, int length, int window, int min_periods, int o_step)
{
    double _, sum_a = 0, sum_b = 0, mean_a, mean_b;
    int i, start = 0, end = window - 1, tmp_window = window;
    min_periods = max(min_periods, 3);
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    for (i = start; i < end; i++, orr+=o_step)
    {
        // orr[i] = NAN;
        *orr = NAN;
        if (ISNAN(arr[i]) || ISNAN(brr[i]))
            tmp_window -= 1;
        else
        {
            _ = arr[i];
            sum_a += _;
            _ = brr[i];
            sum_b += _;
        }
    }

    for (; end < length; start++, end++, orr+=o_step)
    {
        if (ISNAN(sum_a) || ISNAN(sum_b))
        {
            sum_a = 0;
            sum_b = 0;
        }
        if (ISNAN(arr[end]) || ISNAN(brr[end]))
            tmp_window -= 1;
        else
        {
            _ = arr[end];
            sum_a += _;
            _ = brr[end];
            sum_b += _;
        }
        if (tmp_window >= min_periods)
        {
            mean_a = sum_a / tmp_window; // mean_a
            mean_b = sum_b / tmp_window; // mean_b
            _ = 0;
            for (int j = start; j <= end; j++)
            {
                if (!ISNAN(arr[j]) && !ISNAN(brr[j]))
                    _ += (arr[j] - mean_a) * (brr[j] - mean_b);
            }
        }
        else
            _ = NAN;
        // orr[end] = _ / (tmp_window -1);
        *orr = _ / (tmp_window - 1);
        if (ISNAN(arr[start]))
            tmp_window += 1;
        else
        {
            _ = arr[start];
            sum_a -= _;
            _ = brr[start];
            sum_b -= _;
        }
    }    
}

void ts_corr_1d(const double *arr, const double *brr, double *orr, int length, int window, int min_periods, int o_step)
{
    double _, sum_a = 0, sum_a2 = 0, sum_b = 0, sum_b2 = 0, mean_a, mean_a2, mean_b, mean_b2;
    int i, start = 0, end = window - 1, tmp_window = window;
    min_periods = max(min_periods, 3);
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    for (i = start; i < end; i++, orr+=o_step)
    {
        // orr[i] = NAN;
        *orr = NAN;
        if (ISNAN(arr[i]) || ISNAN(brr[i]))
            tmp_window -= 1;
        else
        {
            _ = arr[i];
            sum_a += _;
            sum_a2 += _ * _;

            _ = brr[i];
            sum_b += _;
            sum_b2 += _ * _;
        }
    }

    for (; end < length; start++, end++, orr+=o_step)
    {
        // if (ISNAN(sum_a) || ISNAN(sum_b))
        // {
        //     sum_a = 0;
        //     sum_a2 = 0;
        //     sum_b = 0;
        //     sum_b2 = 0;
        // }
        if (ISNAN(arr[end]) || ISNAN(brr[end]))
            tmp_window -= 1;
        else
        {
            _ = arr[end];
            sum_a += _;
            sum_a2 += _ * _;
            _ = brr[end];
            sum_b += _;
            sum_b2 += _ * _;
        }
        if (tmp_window >= min_periods)
        {   
            mean_a2 = sum_a2 / tmp_window;
            mean_a = sum_a / tmp_window; // mean_a
            mean_a2 -= mean_a * mean_a;    // variance_a
            mean_b2 = sum_b2 / tmp_window;
            mean_b = sum_b / tmp_window; // mean_b
            mean_b2 -= mean_b * mean_b;    // variance_b
            
            _ = 0;
            if (mean_a2 != 0 && mean_b2 != 0) // 方差不能为0
            {
                for (int j = start; j <= end; j++)
                {
                    if (!ISNAN(arr[j]) && !ISNAN(brr[j]))
                        _ += (arr[j] - mean_a) * (brr[j] - mean_b);
                }
                _ /= sqrt(mean_a2 * mean_b2); //  * tmp_window / (tmp_window - 1)
            }
            else
                _ = NAN;
        }
        else
            _ = NAN;
        // orr[end] = _ / tmp_window;
        *orr = _ / tmp_window;
        if (ISNAN(arr[start]) || ISNAN(brr[start]))
            tmp_window += 1;
        else
        {
            _ = arr[start];
            sum_a -= _;
            sum_a2 -= _ * _;
            _ = brr[start];
            sum_b -= _;
            sum_b2 -= _ * _;
        }
    }
}

void ts_reg_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step)
{
    double sum_y, sum_xy, _;
    int j = 0, end = window - 1, tmp_window;
    double sum_x = 0.5 * window * (window + 1); // sum of time from 1 to window
    double divisisor = window * ((double)(window * (window + 1) * (2 * window + 1)) / 6) - sum_x * sum_x; //denominator of slope
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    while (j < end)
    {   
        *orr = NAN;
        orr += o_step;
        j++;
    }

    for (; end < length; end++, orr+=o_step)
    {   
        sum_y = 0;
        sum_xy = 0;
        tmp_window = window;
        for (j = window; j-- != 0;)
        {   
            _ = arr[end - j];
            if (!ISNAN(_))
            {
                sum_y += _;
                sum_xy += (window - j) * _;
            }
            else 
                tmp_window -= 1;
        }
        double b = (window * sum_xy - sum_x * sum_y) / divisisor; // slope
        double a = (sum_y - b * sum_x) / tmp_window; // intercept
        *orr = (tmp_window >= min_periods) ? (a + b * window) : NAN;
    }
}

void ts_tsf_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step)
{
    double sum_y, sum_xy, _;
    int j = 0, end = window - 1, tmp_window;
    double sum_x = 0.5 * window * (window + 1); // sum of time from 1 to window
    double divisisor = window * ((double)(window * (window + 1) * (2 * window + 1)) / 6) - sum_x * sum_x; //denominator of slope
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    while (j < end)
    {   
        *orr = NAN;
        orr += o_step;
        j++;
    }

    for (; end < length; end++, orr+=o_step)
    {   
        sum_y = 0;
        sum_xy = 0;
        tmp_window = window;
        for (j = window; j-- != 0;)
        {   
            _ = arr[end - j];
            if (!ISNAN(_))
            {
                sum_y += _;
                sum_xy += (window - j) * _;
            }
            else 
                tmp_window -= 1;
        }
        double b = (window * sum_xy - sum_x * sum_y) / divisisor; // slope
        double a = (sum_y - b * sum_x) / tmp_window; // intercept
        *orr = (tmp_window >= min_periods) ? (a + b * (window+1)) : NAN;
    }
}

void ts_reg_slope_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step)
{
    double sum_y, sum_xy, _;
    int j = 0, end = window - 1, tmp_window;
    double sum_x = 0.5 * window * (window + 1); // sum of time from 1 to window
    double divisisor = window * ((double)(window * (window + 1) * (2 * window + 1)) / 6) - sum_x * sum_x; //denominator of slope
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    while (j < end)
    {   
        *orr = NAN;
        orr += o_step;
        j++;
    }

    for (; end < length; end++, orr+=o_step)
    {   
        sum_y = 0;
        sum_xy = 0;
        tmp_window = window;
        for (j = window; j-- != 0;)
        {   
            _ = arr[end - j];
            if (!ISNAN(_))
            {
                sum_y += _;
                sum_xy += (window - j) * _;
            }
            else 
                tmp_window -= 1;
        }
        double b = (window * sum_xy - sum_x * sum_y) / divisisor; // slope
        // double a = (sum_y - b * sum_x) / tmp_window; // intercept
        *orr = (tmp_window >= min_periods) ? b : NAN;
    }
}

void ts_reg_intercept_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step)
{
    double sum_y, sum_xy, _;
    int j = 0, end = window - 1, tmp_window;
    double sum_x = 0.5 * window * (window + 1); // sum of time from 1 to window
    double divisisor = window * ((double)(window * (window + 1) * (2 * window + 1)) / 6) - sum_x * sum_x; //denominator of slope
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    while (j < end)
    {   
        *orr = NAN;
        orr += o_step;
        j++;
    }

    for (; end < length; end++, orr+=o_step)
    {   
        sum_y = 0;
        sum_xy = 0;
        tmp_window = window;
        for (j = window; j-- != 0;)
        {   
            _ = arr[end - j];
            if (!ISNAN(_))
            {
                sum_y += _;
                sum_xy += (window - j) * _;
            }
            else 
                tmp_window -= 1;
        }
        double b = (window * sum_xy - sum_x * sum_y) / divisisor; // slope
        double a = (sum_y - b * sum_x) / window; // intercept
        *orr = (tmp_window >= min_periods) ? a : NAN;
    }
}

void ts_wma_1d(const double *arr, double *orr, int length, int window, int min_periods, int o_step){
    double sum=0, sub=0, _, tmp = 0.;
    double divisor;
    int i = 0, j = 0, start = 0, end = window - 1, tmp_window = window;
    if (check_window(window, min_periods, orr, length, o_step))
        return;
    j = 1;
    while( i < end)
    {
        _ = arr[i++];
        if(ISNAN(_))
            tmp_window -= 1;
        else{
            sub += _;
            sum += _ * j;
        }
        j++;
        *orr = NAN;
        orr += o_step;
    }

    for (; end < length; start++, end++, orr+=o_step)
    {
        _ = arr[end];
        if(ISNAN(_))
            tmp_window -= 1;
        else{
            sub += _;
            sub -= tmp;
            sum += _ * window;
        }
        divisor = (tmp_window * (tmp_window + 1)) >> 1;
        *orr = (tmp_window >= min_periods) ? (sum / divisor) : NAN;
        _ = arr[start];
        if(ISNAN(_))
        {
            tmp = 0;
            tmp_window += 1;
        }
        else
        {
            tmp = _;        
            sum -= sub;
        }
    }
}

