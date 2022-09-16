# `RollingStatistics`: Fast & Efficient Calculation of Rolling Mean/Variance/Maximum/Rank etc. for C++ and Python

This is a lightweight library for calculation of rolling statistics, mainly in two ways:

(1) Inplace calculation over an n-dimensional array along a specified axis, with a fixed window. The array should be represented by a pointer, and may have custom storage order (row-major, column-major or some custom strides). This would include `numpy.ndarray` and C arrays, for example.

(2) Manual calculation with a flexible window, via `push()` and `pop()` operations.

Let's get right into the examples.

## A Starter Example in C++
```cpp
#include <iostream>
#include "src/rolling_statistics.hpp"

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

## A Starter Example in Python
```py
import numpy as np
import rolling_statistics_py as rsp

# unstructured data, work exactly the same as in C++.

# structured example 1: same array as C++, but we now use a wrapper function roll_ndarray_float().
arr = np.array([[2.0, 3.0, 1.0],
               [3.0, 3.5, np.nan],
               [np.nan, 4.0, 2.0],
               [-3.0, np.nan, np.nan]], dtype='float32')
rolling_mean = rsp.RollingMean_float()

rsp.roll_ndarray_float(arr, rolling_mean, axis=0, window=3, min_periods=2)
print(arr)  # same result as C++.

# structured example 2:
# suppose we have 20 days of stock returns, 5000 stocks and 240 minutes per day. This array is about 100MB in float32.
# now we want the 5-minute rolling rank of returns in each minute, with at least 3 valid entries in the window.
np.random.seed(0)
arr = np.random.randn(20, 240, 5000).astype('float32')  # explicitly cast to float32
print(arr[10, :20, 42])  # check the first 20 minutes of stock no.43 on day 11

# [-0.31888023 -0.19876899 -0.668215    1.2044029   1.0545355  -1.6606108
#  -1.1592734   0.8667814   0.51651764 -0.17564432 -0.16574599  0.92819685
#  -0.27120432 -0.6692324   2.0230536   0.17266187 -1.3617305   0.09074531
#   0.37932783 -0.76033247]

rolling_rank = rsp.RollingRank_float()
rsp.roll_ndarray_float(arr, rolling_rank, axis=1, window=5, min_periods=3)
print(arr[10, :20, 42])

# [nan nan  0.  3.  3.  0.  1.  2.  2.  2.  2.  4.  0.  0.  4.  2.  0.  2.
#   3.  1.]
```

## Prerequisites

For both C++ and Python, you will need a distribution of `g++` that supports at least C++11. `g++` is mandatory, as this library uses `__gnu_pbds::tree` from libstdc++, which is part of the GNU implementation for C++ standard library.

Additionally for Python, you need to install `Pybind11` in your Python environment, and it is highly recommended to create a virtual environment (either pure python, or an anaconda one).

Installing from pip in the virtual environment is easy:

`$ .../env_name/Scripts/pip install pybind11`

Or with anaconda, simply run the following commands with the virtual environment activated:

`$ conda install pybind11`

## Installation

For C++, no installation is needed, the only file you need is `src/rolling_statistics.hpp`. For Python, two installation methods are provided:

### setuptools

This method is the easiest, just run the following commands in the project root folder:

```
$ cd src
$ python setup.py install
```

There is one drawback though: it is not possible to specify which compiler to use. Therefore, on Windows, there is a high chance that the MSVC compiler will be used by default. If you see some errors along the lines of `"ext/pb_ds/assoc_container.hpp": No such file or directory`, that would be the case, and you need to use the second method. If successful, `setup.py` will automatically copy a `.egg` file to your project root folder. This file is essentially a zip file, and the internal contents depend on your OS: for Windows it will be `.pyd`, for Linux it will be `.so`. In any case, Python wraps it in a neat manner, and you can directly import with `import rolling_statistics_py` without extracting it.

### makefile

For this method, you will first need to install `cmake`. Then, run the following commands in the project root directory:

For Windows (MinGW),

```
$ mkdir build
$ cd build
$ cmake .. -G "MinGW Makefiles" -DPYTHON_LIBRARY_DIR=".../env_name/Lib/site-packages" -DPYTHON_EXECUTABLE=".../env_name/python.exe" -Dpybind11_DIR=".../env_name/Lib/site-packages/pybind11/share/cmake/pybind11"
$ MinGW32-make
$ MinGW32-make install
```

For Linux,

```
$ mkdir build
$ cd build
$ cmake .. -DPYTHON_LIBRARY_DIR=".../env_name/Lib/site-packages" -DPYTHON_EXECUTABLE=".../env_name/python.exe" -Dpybind11_DIR=".../env_name/Lib/site-packages/pybind11/share/cmake/pybind11"
$ make
$ make install
```

Now, you should find a file with name similar to `rolling_statistics_py.cp36-win_amd64.pyd` in your `.../env_name/Lib/site-packages` folder, and you are ready to go. To test whether the library is properly installed, you can run `usage_example_python.py` in your virtual environment.

If somehow neither of the methods works, you can leave a message or check on the [Pybind11 Documentation](https://pybind11.readthedocs.io/en/stable/compiling.html#build-systems) for other methods.

## Usage Documentation: Interfaces

All classes are derived from `RS::RollingStatistics` and provide the same interfaces, for both C++ and Python (except `roll_ndarray()`, which you will see).

### RS::RollingStatistics<value_type>::RollingStatistics
```cpp
RollingStatistics(bool skip_nan=true)
```

The constructor. It accepts one parameter `skip_nan`. If this is true (by default), then computation of the rolling statistics will skip all NaN values inside the window, as seen in the unstructured example of C++. Otherwise, any NaN value within the window will propagate, i.e. `compute()` will return `NAN`.

### RS::RollingStatistics<value_type>::clear
```cpp
void clear()
```

Clears all internal data. This function is also called by the constructor.

### RS::RollingStatistics<value_type>::front
```cpp
void value_type front()
```

Returns the oldest element in the current window. Unlike in STL containers, this method does not return a reference, as the stored values should not be changed.

### RS::RollingStatistics<value_type>::push
```cpp
void push(const value_type& val)
```

Pushes a value into the internal data structures.

### RS::RollingStatistics<value_type>::pop
```cpp
void pop()
```

Pops the oldest value in the current window from the internal data structures.

### RS::RollingStatistics<value_type>::size
```cpp
size_t size()
```

Returns the number of elements in the current window.

### RS::RollingStatistics<value_type>::size_nan
```cpp
size_t size_nan()
```

CHANGE THIS
Returns the number of NaN values in the current window.

### RS::RollingStatistics<value_type>::size_notnan
```cpp
size_t size_notnan()
```

Returns the number of non-NaN values in the current window.

### RS::RollingStatistics<value_type>::compute
```cpp
value_type compute()
```

Returns the target rolling statistics (mean, variance, etc.) calculated from the current window.

### RS::RollingStatistics<value_type>::roll_ndarray
```cpp
roll_ndarray(value_type* ptr_arr, const std::vector<size_t>& shape, size_t axis, size_t window, size_t min_periods, std::vector<size_t> strides={})
```

Performs inplace `compute()` along the specified axis of the given array.

`ptr_arr`: Pointer to the target array, and must *not* be located in a read-only section of the memory (for example, a `const` array compiled by `gcc`).

`shape`: Shape of the target array. The `size()` of this vector would be the number of dimensions.

`axis`: The axis along which to perform the computation. The function will process groups of `shape[axis]` number of cells at a time, before calling `clear()` and moving on to the next group.

`window`: The maximum size of the rolling window. The first `window - 1` values of each group will use a window size the same as their position (`1, 2, ... window - 1`), all values after will use window size `window`.

`min_periods`: The minimum requirement for non-NaN values in the current window to perform computarion. If not met, the current cell will be replaced with `NAN` instead. If this value is positive, the first `min_periods` cells of each group will always be set to `NAN`.

`strides`: The number of positions (not bytes, unlike in NumPy) to skip to reach the next cell in each dimension, *Leave empty unless absolutely necessary*. This is meant as an interface to `numpy.ndarray`, which uses strides to determine the expansion order of an n-dimensional array, or even skip some parts of the memory to achieve some advanced indexing. Arrays in C++ always use row-major order, which is the default behavior for this parameter. As the [NumPy official documentary](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html) mentions, meddling with strides should be done with extreme care. We have added an extra protection to prevent the pointer from going out of bounds of the array, should you somehow end up in a situation to utilize this parameter.

Note: As seen in the starter example, this interface is not provided in Python, use instead the following wrapper function. This is due to the lack of pointer variables in python, the function then resorts to 'fetching' the pointer imbedded in a `numpy.ndarray`. Note that the array must not be a temporary object (or in C++ terms, an rvalue).

```py
roll_ndarray(ndarray, rolling_statistics, axis, window, min_periods)
```

## Usage Documentation: Classes

### RS::RollingMean<value_type>

Yields rolling mean for computation. Uses `std::queue` as internal data structure. `O(n)` time and `O(window)` space complexity, where `n` id the size of the entire array (for a structured array, it is the production of all values in `shape`), and `window` is the maximum size of the window throughout the lifetime of the object (fixed for a structured array).

$$\frac{ \Sigma_{i \in I}X_i}{|I|}$$

### RS::RollingVariance<value_type>

Yields (biased) rolling variance for computation. Uses `std::queue` as internal data structure. `O(n)` time and `O(window)` space complexity.

### RS::RollingSkewness<value_type>


### RS::RollingMaximum<value_type>


### RS::RollingMinimum<value_type>


### RS::RollingRank<value_type>


### RS::RollingQuantile<value_type>


## Q&A

Q: I applied `roll_ndarray()` to a numpy array but the array is not changed, why?

A: It is highly likely the datatypes do not match. For example, numpy creates arrays in `np.float64` by default. If you create a `RollingMean_float` and use it in `roll_ndarray()`, Python will first perform a shallow copy of the array, casting it to float, then pass it to C++. In this case, `roll_ndarray()` will only change the temporary array, which is soon destroyed. To resolve this, either cast the original array into the appropriate dtpye using `np.astype()`, or use the appropriate function, e.g. `RollingMean_double` for the above exmaple.

## Future Updates

(1) Optimize virtual functions.

(2) Implement support for multiple windows in `RollingMomentStatistics`, and some two-sample t-tests.
