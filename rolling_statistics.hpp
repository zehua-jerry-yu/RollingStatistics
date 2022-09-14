#ifndef ROLLING_STATISTICS_HPP
#define ROLLING_STATISTICS_HPP

/**
 * @file rolling_statistics.hpp
 * @author Zehua Yu (zehua.yu@columbia.edu)
 * @version 1.0
 * @date 2022-09-12
 * @brief
 */

/*  Inheritance:
 *                                    RollingStatistics
 *                             /       |          |          \
 *         RollingMomentStatics   RollingMax  RollingQuantile  ...
 *           /        |         \
 *    RollingMean  RollingVariance  ...
 *
 * */

#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <queue>
#include <utility>
#include <algorithm>
#include <unordered_map>
#include <initializer_list>


namespace RS{

const double EPSILON = 1.0e-16;

template <typename D>
class RollingStatistics{
    /* The base class. */
protected:
    bool skip_nan = true;  // whether to skip NaN values in a window, or to propagate them.
    size_t num_values_nan = 0;  // number of NaNs in the current window.
    size_t num_values_notnan = 0;  // number of non-NaNs in the current window.
    virtual D compute_aux() = 0;  // compute target statistics.
public:
    static const std::string name;  // prefix for name of class in Python
    virtual void clear() = 0;
    // accessor functions
    inline size_t size() const { return num_values_nan + num_values_notnan; }
    inline const size_t& size_nan() const { return num_values_nan; }
    inline const size_t& size_notnan() const { return num_values_notnan; }
    virtual D front() = 0;
    virtual void push(const D& value) = 0;
    virtual void pop() = 0;
    D compute() {
        if (num_values_notnan == 0 || (!skip_nan && num_values_nan > 0)) {
            return NAN;
        } else {
            return compute_aux();
        }
    }

    void roll_ndarray(D* ptr_arr, const std::vector<size_t>& shape, size_t axis, size_t window, size_t min_periods, std::vector<size_t> strides={}) {
        /* inplace rolling, accepts one pointer. */
        size_t ndim = shape.size();
        assert(ndim > 0);
        assert(strides.empty() || strides.size() == ndim);

        // calculate the size of the array
        size_t size_arr = 1;
        for (size_t i = 0; i != ndim; ++i){
            size_arr *= shape[i];
        }
        // initialise strides to c-style
        if (strides.empty()){
            size_t stride = 1;
            for (size_t i = 0; i != ndim; ++i){
                if (i > 0){ stride *= shape[ndim - i]; }
                strides.push_back(stride);
            }
            std::reverse(strides.begin(), strides.end());
        }
        // above code is different for cpp and python

        std::vector<size_t> indices(ndim);  // stores indices for each axis
        size_t scalar_index;  // index for flattened array
        // perform ndim layers of loop (innermost layer is axis 'axis')
        size_t current_axis;
        if (ndim == 1) {
            current_axis = 0;
        }
        else {
            current_axis = (axis == 0 ? 1 : 0);  // start from the smallest axis that is not 'axis'
        }
        while (true) {
            clear();
            scalar_index = 0;
            indices[axis] = 0;
            for (size_t i = 0; i != ndim; ++i) {
                scalar_index += indices[i] * strides[i];
            }
            assert(scalar_index < size_arr);  // prevent out of bounds
            for (indices[axis] = 0; indices[axis] != shape[axis]; ++indices[axis]) {
                push(*(ptr_arr + scalar_index));
                if (indices[axis] >= window) {
                    pop();
                }
                if (size_notnan() >= min_periods) {
                    *(ptr_arr + scalar_index) = compute();
                }
                else {
                    *(ptr_arr + scalar_index) = NAN;
                }
                scalar_index += strides[axis];
            }
            assert(indices[axis] > 0);
            --indices[axis];
            // go to the next layer until it's not full, then increment it and go back to the first layer.
            if (indices[current_axis] == shape[current_axis] - 1) {
                while (indices[current_axis] == shape[current_axis] - 1) {
                    indices[current_axis] = 0;
                    ++current_axis;
                    if (current_axis == axis) {
                        ++current_axis;
                    }
                }
                if (current_axis < ndim) {
                    ++indices[current_axis];
                }
                else {
                    break;
                }
                current_axis = (axis == 0 ? 1 : 0);
            }
            else {
                ++indices[current_axis];
            }
        }
    }

};


template <typename D>
class RollingMomentStatistics: public RollingStatistics<D> {
    /* An abstract class for rolling moment statistics, e.g. rolling mean. */
protected:
    std::vector<D> unnormalized_moments;  // one or more variables to maintain, e.g. rolling sum of x_i or x_i^2.
    std::vector<std::queue<D>> vecs_in_window;  // unnormalized_moments.size() number of queues of x_i^j, one queue for each j.
    size_t num_moments = 0;
    inline const std::vector<D>& get_moments() const { return this->unnormalized_moments; }
public:
    RollingMomentStatistics(bool skip_nan_, int num_moments_){
        this->skip_nan = skip_nan_;
        this->num_moments = num_moments_;
        clear();
    }
    void clear() {
        /* can be manually called or called by the constructor */
        this->unnormalized_moments = std::vector<D>(this->num_moments, 0);
        this->vecs_in_window = std::vector<std::queue<D>>(this->num_moments);
        this->num_values_nan = 0;
        this->num_values_notnan = 0;
    }
    D front() {
        /* return the next (original x, not x^2 etc.) value that will be popped. */
        assert(!this->vecs_in_window[0].empty());
        return this->vecs_in_window[0].front();
    }
    virtual void pop() {
        /* most derived classes will use this, but some such as zscore do not. */
        for (size_t index = 0; index != this->num_moments; ++index) {
            pop_aux(index);
        }
    }
    void push_aux(D value, size_t index) {
        /* add a new value to the maintained window */
        this->vecs_in_window[index].push(value);
        if (!std::isnan(value)) {
            this->unnormalized_moments[index] += value;
            if (index == 0) {  // if num_moments > 1, multiple add() will be called for one cell
                ++this->num_values_notnan;
            }
        }
        else {
            if (index == 0) {
                ++this->num_values_nan;
            }
        }
    }
    void pop_aux(size_t index) {
        /* remove a value from the maintained window, which should have been pushed before. */
        assert(!this->vecs_in_window[index].empty());
        const D& value = this->vecs_in_window[index].front();
        if (!std::isnan(value)) {
            this->unnormalized_moments[index] -= value;
            if (index == 0) { --this->num_values_notnan; }
        }
        else {
            if (index == 0) { --this->num_values_nan; }
        }
        this->vecs_in_window[index].pop();
    }
};


template <typename D>
class RollingMean : public RollingMomentStatistics<D> {
    /* unnormalized_moments[0] stores \Sum{x_i} */
protected:
    D compute_aux() {
        D n = static_cast<D>(this->num_values_notnan);
        return this->get_moments()[0] / n;
    }
public:
    static const std::string name;
    void push(const D& value) {
        this->push_aux(value, 0);
    }
    explicit RollingMean(bool skip_nan_=true): RollingMomentStatistics<D>(skip_nan_, 1){}
};
template <typename D>
const std::string RollingMean<D>::name = "RollingMean";


template <typename D>
class RollingVariance : public RollingMomentStatistics<D> {
    /* unnormalized_moments[0] stores \Sum{x_i}, unnormalized_moments[1] stores \Sum{x_i^2} */
protected:
    D compute_aux() {
        // \Sum{(x_i - x_mean)^2} / n = \Sum{x_i^2} / n - x_mean^2
        D n = static_cast<D>(this->num_values_notnan);
        const std::vector<D>& moments_ = this->get_moments();
        D x_mean = moments_[0] / n;
        return moments_[1] / n - x_mean * x_mean;
    }
public:
    static const std::string name;
    void push(const D& value) {
        this->push_aux(value, 0);
        this->push_aux(value * value, 1);
    }
    explicit RollingVariance(bool skip_nan_=true): RollingMomentStatistics<D>(skip_nan_, 2){}
};
template <typename D>
const std::string RollingVariance<D>::name = "RollingVariance";


template <typename D>
class RollingSkewness : public RollingMomentStatistics<D> {
    /* unnormalized_moments: \Sum{x_i}, \Sum{x_i^2}, \Sum{x_i^3} */
protected:
    D compute_aux() {
        /*
        \Sum{(x_i - x_mean)^3} / n
        = [\Sum{x_i^3 - 3 * x_i^2 * x_mean + 3 * x_i * x_mean^2 - x_mean^3}] / n
        =  \Sum{x_i^3} / n - 3 * \Sum{x_i^2} / n * x_mean + 2 * x_mean^3
        finally divie by sigma^3
        */
        D n = static_cast<D>(this->num_values_notnan);
        const std::vector<D>& moments_ = this->get_moments();
        D x_mean = moments_[0] / n;
        D x_var = moments_[1] / n - x_mean * x_mean;
        if (x_var < EPSILON) {
            return NAN;
        }
        else {
            return (moments_[2] / n - 3 * moments_[1] / n * x_mean + 2 * x_mean * x_mean * x_mean) / pow(x_var, static_cast<D>(1.5));
        }
    }
public:
    static const std::string name;
    void push(const D& value) {
        this->push_aux(value, 0);
        this->push_aux(value * value, 1);
        this->push_aux(value * value * value, 2);
    }
    explicit RollingSkewness(bool skip_nan_=true): RollingMomentStatistics<D>(skip_nan_, 3){}
};
template <typename D>
const std::string RollingSkewness<D>::name = "RollingSkewness";


template <typename D>
class RollingZScore : public RollingMomentStatistics<D> {
    /* unnormalized_moments[0]~[2] store \Sum{x_i}, x_i, \Sum{x_i^2} */
protected:
    D compute_aux() {
        D n = static_cast<D>(this->num_values_notnan);
        const std::vector<D>& moments_ = this->get_moments();
        D x = moments_[1];
        D x_mean = moments_[0] / n;
        D x_var = moments_[2] / n - x_mean * x_mean;
        if (x_var < EPSILON) {
            return NAN;
        } else {
            return (x - x_mean) / sqrt(x_var);
        }
    }
public:
    static const std::string name;
    void push(const D& value) {
        this->push_aux(value, 0);
        if (this->vecs_in_window[1].size() >= 1) {  // maintain a short window of 1
            this->pop_aux(1);
        }
        this->push_aux(value, 1);
        this->push_aux(value * value, 2);
    }
    void pop() {  // also tricky part of zscore
        this->pop_aux(0);
        this->pop_aux(2);
    }
    explicit RollingZScore(bool skip_nan_=true): RollingMomentStatistics<D>(skip_nan_, 3){}
};
template <typename D>
const std::string RollingZScore<D>::name = "RollingZScore";


template <typename D>
class RollingMax : public RollingStatistics<D>{
    /* uses a std::deque, see same question in leetcode for explanation. */
protected:
    std::queue<D> vals_in_window;
    std::deque<D> maximums;
    D compute_aux(){
        assert(!maximums.empty());
        return maximums.front();
    }
public:
    explicit RollingMax(bool skip_nan_){ this->skip_nan = skip_nan_; clear(); }
    static const std::string name;
    void clear() {
        /* can be manually called or called by the constructor */
        vals_in_window = std::queue<D>();
        maximums = std::deque<D>();
        this->num_values_nan = 0;
        this->num_values_notnan = 0;
    }
    D front(){
        assert(!vals_in_window.empty());
        return vals_in_window.front();
    }
    void push(const D& value){
        vals_in_window.push(value);
        if (std::isnan(value)){
            ++this->num_values_nan;
        } else {
            while (!maximums.empty() && maximums.back() < value){ maximums.pop_back(); }
            maximums.push_back(value);
            ++this->num_values_notnan;
        }
    }
    void pop(){
        D value = front();
        vals_in_window.pop();
        if (std::isnan(value)){
            --this->num_values_nan;
        } else {
            if (value == maximums.front()){ maximums.pop_front(); }
            --this->num_values_notnan;
        }
    }
};
template <typename D>
const std::string RollingMax<D>::name = "RollingMax";


template <typename D>
class RollingMin : public RollingStatistics<D>{
protected:
    std::queue<D> vals_in_window;
    std::deque<D> minimums;
    D compute_aux(){
        assert(!minimums.empty());
        return minimums.front();
    }
public:
    explicit RollingMin(bool skip_nan_){ this->skip_nan = skip_nan_; clear(); }
    static const std::string name;
    void clear() {
        /* can be manually called or called by the constructor */
        vals_in_window = std::queue<D>();
        minimums = std::deque<D>();
        this->num_values_nan = 0;
        this->num_values_notnan = 0;
    }
    D front(){
        assert(!vals_in_window.empty());
        return vals_in_window.front();
    }
    void push(const D& value){
        vals_in_window.push(value);
        if (std::isnan(value)){
            ++this->num_values_nan;
        } else {
            while (!minimums.empty() && minimums.back() > value){ minimums.pop_back(); }
            minimums.push_back(value);
            ++this->num_values_notnan;
        }
    }
    void pop(){
        D value = front();
        vals_in_window.pop();
        if (std::isnan(value)){
            --this->num_values_nan;
        } else {
            if (value == minimums.front()){ minimums.pop_front(); }
            --this->num_values_notnan;
        }
    }
};
template <typename D>
const std::string RollingMin<D>::name = "RollingMin";


}  // namespace RS
#endif
