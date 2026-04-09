#include "get_potential_difference.hpp"
#include "get_potential_difference.cpp"
#include <pybind11/pybind11.h>
#include<pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(c_imp_get_potential_difference, m){
    m.doc() = "test";
    m.def("get_potential_difference", &get_potential_difference, "Gets the potential difference between the candidate and current actions");
}