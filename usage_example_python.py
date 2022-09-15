"""
Some simple examples to showcase the use of rolling_statistics in Python.
"""

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

# structured exmaple 2:
# suppose we have 20 days of stock returns, 5000 stocks and 240 minutes per day. This array is about 100MB in float32.
# now we want the 5-minunte rolling quantile of returns in each minute, with at least 3 valid entries in the window.
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
