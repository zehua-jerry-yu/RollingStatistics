import numpy as np
import rolling_statistics_py as rs

# unstructured data, work exactly the same as in C++.

# structured: sample example, function parameters changed in Python.
arr = np.array([[2.0, 3.0, 1.0],
               [3.0, 3.5, np.nan],
               [np.nan, 4.0, 2.0],
               [-3.0, np.nan, np.nan]])
rolling_mean = rs.RollingMean_double()
rolling_mean.roll_ndarray(arr=arr, axis=0, window=3, min_periods=2)
print(arr)  # same result as C++.

# another structured example:
# suppose we have 20 days of stock returns, 5000 stocks and 240 minutes per day. This array is about 100MB in float32.
# now we want the 5-minunte rolling quantile of returns in each minute, with at least 3 valid entries in the window.

