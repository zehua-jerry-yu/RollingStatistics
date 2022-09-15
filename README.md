# `RollingStatistics`: Fast & Efficient Calculation of Rolling Mean/Variance/Maximum/Rank etc. for C++ and Python

\[Note: this readme is not finished.\]

This is a lightweight library for calculation of rolling statistics, mainly in two ways:

(1) Inplace calculation over an n-dimensional array along a particular axis, with a fixed window. The array should be represented by a pointer, and may have custom storage order (row-major, column-major or some custom strides). This would include `numpy.ndarray` and C arrays, for example.

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
    rolling_mean.roll_ndarray(&arr[0][0], shape, 0, 3, 2);  // axis=0, window=3, min_periods=2. stride uses default c-style.
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

## Example in Python
```py
import numpy as np
import rolling_statistics_py as rs

arr = np.array([[2.0, 3.0, 1.0],
               [3.0, 3.5, np.nan],
               [np.nan, 4.0, 2.0],
               [-3.0, np.nan, np.nan]])
rolling_mean = rs.RollingMean_double()
rolling_mean.roll_ndarray(arr=arr, axis=0, window=3, min_periods=2)
print(arr)  # same result

```
## Q&A

Q: I applied `roll_ndarray()` to a numpy array but the array is not changed, why?

A: It is highly likely the datatypes do not match. For example, numpy creates arrays in `np.float64` by default. If you create a `RollingMean_float` and use it in `roll_ndarray()`, Python will first perform a shallow copy of the array, casting it to float, then pass it to C++. In this case, `roll_ndarray()` will only changed the temporary array. To resolve this, either cast the original array into the appropriate dtpye using `np.astype()`, or use the appropriate function, e.g. `RollingMean_double` for the above exmaple.

## Future Updates

(1) Optimize virtual functions.

(2) Implement support for multiple windows in `RollingMomentStatistics`, and some two-sample t-tests.

(3) May switch to use order statistics tree from `libstdc++` for O(logm) order statistics operation and rank operation. Currently Fenwick Tree is preferred due to its lightweight.
