# `RollingStatistics`: Fast & Efficient Calculation of Rolling Mean/Variance/Maximum/Quantile etc. for C++ and Python

This is a lightweight library for calculation of rolling statistics, mainly in two ways:

(1) Inplace calculation over an n-dimensional array over a particular axis, with a fixed window. The array should be represented by a pointer, and may have custom storage order (row-major, column-major or some custom strides). This would include `numpy.ndarray` and C arrays, for example.

(2) Manual calculation with a flexible window, via `push()` and `pop()` operations.

Let's get right into the examples.

## A Starter Example in C++
```cpp
#include <iostream>
#include "rolling_statistics.hpp"

int main() {
    // default constructor: skip_nan=true
    RS::RollingMean<float> rolling_mean;

    // example for unstructured data
    // suppose we get 3, 2, 0 and 1 entries on the first 4 days. calculate a 2-day rolling mean.
    rolling_mean.push(1.0);
    rolling_mean.push(2.0);
    rolling_mean.push(3.0);
    std::cout << "day1: " << rolling_mean.compute() << std::endl;  // 2.0
    rolling_mean.push(NAN);
    rolling_mean.push(4.0);
    std::cout << "day2: " << rolling_mean.compute() << std::endl;  // 2.5
    for (int i = 0; i != 3; ++i) { rolling_mean.pop(); }
    std::cout << "day3: " << rolling_mean.compute() << std::endl;  // 4.0
    for (int i = 0; i != 2; ++i) { rolling_mean.pop(); }
    rolling_mean.push(NAN);
    std::cout << "day4: " << rolling_mean.compute() << std::endl;  // NAN

    // example for structured data (n-dimensioanal arrays). we want a 3-day rolling mean, with at least 2 valid entries.
    // suppose there are 3 entities (e.g. stocks) and 4 days of data.
    // rolling_mean.clear();  // no need, will be called automatically by roll_ndarray()
    float arr[4][3] = {{2.0, 3.0, 1.0},
                       {3.0, 3.5, NAN},
                       {NAN, 4.0, 2.0},
                       {-3.0, NAN, NAN}};
    const std::vector<size_t> shape = {4, 3};
    const std::vector<size_t> strides = {3, 1};  // row-major strides (of values, not bytes)
    rolling_mean.roll_ndarray(&arr[0][0], shape, strides, 0, 3, 2);  // axis=0, window=3, min_periods=2
    std::cout << "arr has changed to:" << std::endl;
    for (size_t i = 0; i != shape[0]; ++i){
        for (size_t j = 0; j != shape[1]; ++j){
            std::cout << arr[i][j] << " ";
        }
        std::cout << std::endl;
    }
    /* result:
     * nan  nan  nan
     * 2.5  3.25 nan
     * 2.5  3.5  1.5
     * 0    3.75 nan
     */
    system("pause");
    return 0;
}

```
