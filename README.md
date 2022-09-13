# `RollingStatistics`: Fast & Efficient Calculation of Rolling Mean/Variance/Maximum/Quantile etc. for C++ and Python



## A Starter Example in C++
```cpp
#include <iostream>
#include "rolling_statistics.hpp"

int main() {
    // default constructor: skip_nan=true
    RS::RollingMean<float> rolling_mean;

    // example for unstructured data
    rolling_mean.push(2.0);
    rolling_mean.push(3.0);
    rolling_mean.push(4.0);
    std::cout << "value: " << rolling_mean.compute() << std::endl;  // 3.0
    rolling_mean.push(NAN);
    rolling_mean.pop();
    std::cout << "value: " << rolling_mean.compute() << std::endl;  // 3.5

    // example for structured data (n-dimensioanal arrays). roll_ndarray() will automatically call clear().
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
     * nan nan nan
     * 2.5 3.25 nan
     * 2.5 3.5 1.5
     * 0 3.75 nan
     */
    system("pause");
    return 0;
}

```
