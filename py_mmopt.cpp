#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#define EXPORT_TENSORND_TO_PYBIND11
#include "include/tensorND.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j)
{
    return i + j;
}



namespace py = pybind11;

PYBIND11_MODULE(mmopt, m)
{
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: mmopt

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";


    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def(
        "subtract", [](int i, int j)
        { return i - j; },
        R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    tensorND<float>::bind2py(m);

    m.def("h", [](tensorND<float> & t, float x) {
        t.for_each([&](size_t idx, int * coord){
            t(coord) = x;
            return true;
        });
    });

    m.def("g", []() {
        tensorND<float> t({4,5}, false);
        t.for_each([&](size_t idx, int * coord){
            t(coord) = idx;
            return true;
        });
        return t;
    });
    m.def("g2", []() {
        tensorND<float> t({4,5}, false);
        t.for_each([&](size_t idx, int * coord){
            t(coord) = idx;
            return true;
        });
        return t.to_array();
    });

    m.def("f", []() {
        // Allocate and initialize some data; make this big so
        // we can see the impact on the process memory use:
        constexpr size_t size = 100*1000*1000;
        double *foo = new double[size];
        for (size_t i = 0; i < size; i++) {
            foo[i] = (double) i;
        }
        std::cerr << "created memory @ " << reinterpret_cast<void*>(foo) << "\n";
        // Create a Python object that will free the allocated
        // memory when destroyed:
        py::capsule free_when_done(foo, [](void *f) {
            double *foo = reinterpret_cast<double *>(f);
            std::cerr << "Element [0] = " << foo[0] << "\n";
            std::cerr << "freeing memory @ " << f << "\n";
            delete[] foo;
        });

        return py::array_t<double>(
            {100, 1000, 1000}, // shape
            {1000*1000*8, 1000*8, 8}, // C-style contiguous strides for double
            foo, // the data pointer
            free_when_done); // numpy array references this parent
    });

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}