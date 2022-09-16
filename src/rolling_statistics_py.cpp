#include <math.h>  // must import before pybind11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "rolling_statistics.hpp"
namespace py = pybind11;


template <typename D>
void roll_ndarray(py::array_t<D> arr, RS::RollingStatistics<D>& rs, size_t axis, size_t window, size_t min_periods){
    py::buffer_info info_arr = arr.request();
    D* ptr_arr = static_cast<D*>(info_arr.ptr);
    std::vector<size_t> shape;
    for (py::ssize_t& s: info_arr.shape){
        shape.push_back(static_cast<size_t>(s));
    }
    std::vector<size_t> strides;
    for (py::ssize_t& s: info_arr.strides){
        strides.push_back(static_cast<size_t>(s / info_arr.itemsize));
    }
    rs.roll_ndarray(ptr_arr, shape, axis, window, min_periods, strides);
}


template <typename D, class Class>
void declare_array_RollingStatistics(py::module& m, const std::string& typestr) {
    /*  A helper function to expose derived classes to Python.  */
    std::string pyclass_name = Class::name + std::string("_") + typestr;
    // tell pybind11 that DerivedClass<D> extends BaseClass<D>
    py::class_<Class, RS::RollingStatistics<D>>(m, pyclass_name.c_str())
        .def(py::init<bool>(), py::arg("skip_nan")=true)
        .def("clear", &Class::clear)
        .def("size_nan", &Class::size_nan)
        .def("size_notnan", &Class::size_notnan)
        .def("front", &Class::front)
        .def("push", &Class::push, py::arg("val"))
        .def("pop", &Class::pop)
        .def("compute", &Class::compute);
}


template <typename D, class Class>
void declare_array_RollingRank(py::module& m, const std::string& typestr) {
    std::string pyclass_name = Class::name + std::string("_") + typestr;
    py::class_<Class, RS::RollingStatistics<D>>(m, pyclass_name.c_str())
        .def(py::init<bool, bool>(), py::arg("skip_nan")=true, py::arg("normalize")=false)
        .def("clear", &Class::clear)
        .def("size_nan", &Class::size_nan)
        .def("size_notnan", &Class::size_notnan)
        .def("front", &Class::front)
        .def("push", &Class::push, py::arg("val"))
        .def("pop", &Class::pop)
        .def("compute", &Class::compute);
}


template <typename D, class Class>
void declare_array_RollingOrderStatistics(py::module& m, const std::string& typestr) {
    std::string pyclass_name = Class::name + std::string("_") + typestr;
    py::class_<Class, RS::RollingStatistics<D>>(m, pyclass_name.c_str())
        .def(py::init<D, bool, bool>(), py::arg("order"), py::arg("skip_nan")=true, py::arg("normalize")=false)
        .def_readwrite("order", &Class::order)
        .def("clear", &Class::clear)
        .def("size_nan", &Class::size_nan)
        .def("size_notnan", &Class::size_notnan)
        .def("front", &Class::front)
        .def("push", &Class::push, py::arg("val"))
        .def("pop", &Class::pop)
        .def("compute", &Class::compute);
}



PYBIND11_MODULE(rolling_statistics_py, m) {
    // we will only provide float types because NAN cannot be cast to int.
    m.def("roll_ndarray_float", &roll_ndarray<float>, py::arg("arr"), py::arg("rs"), py::arg("axis"), py::arg("window"), py::arg("min_periods"));
    m.def("roll_ndarray_double", &roll_ndarray<double>, py::arg("arr"), py::arg("rs"), py::arg("axis"), py::arg("window"), py::arg("min_periods"));

    // declare base class - this simply exposes it to Python, it's impossible to
    // construct a BaseClass_float in Python since no constructor is provided
    py::class_<RS::RollingStatistics<float>>(m, "RollingStatistics_float");
    py::class_<RS::RollingStatistics<double>>(m, "RollingStatistics_double");

    declare_array_RollingStatistics<float, RS::RollingMean<float>>(m, std::string("float"));
    declare_array_RollingStatistics<double, RS::RollingMean<double>>(m, std::string("double"));
    declare_array_RollingStatistics<float, RS::RollingVariance<float>>(m, std::string("float"));
    declare_array_RollingStatistics<double, RS::RollingVariance<double>>(m, std::string("double"));
    declare_array_RollingStatistics<float, RS::RollingSkewness<float>>(m, std::string("float"));
    declare_array_RollingStatistics<double, RS::RollingSkewness<double>>(m, std::string("double"));
    declare_array_RollingStatistics<float, RS::RollingZScore<float>>(m, std::string("float"));
    declare_array_RollingStatistics<double, RS::RollingZScore<double>>(m, std::string("double"));
    declare_array_RollingStatistics<float, RS::RollingMax<float>>(m, std::string("float"));
    declare_array_RollingStatistics<double, RS::RollingMax<double>>(m, std::string("double"));
    declare_array_RollingStatistics<float, RS::RollingMin<float>>(m, std::string("float"));
    declare_array_RollingStatistics<double, RS::RollingMin<double>>(m, std::string("double"));
    declare_array_RollingRank<float, RS::RollingRank<float>>(m, std::string("float"));
    declare_array_RollingRank<double, RS::RollingRank<double>>(m, std::string("double"));
    declare_array_RollingOrderStatistics<float, RS::RollingOrderStatistics<float>>(m, std::string("float"));
    declare_array_RollingOrderStatistics<double, RS::RollingOrderStatistics<double>>(m, std::string("double"));
}
