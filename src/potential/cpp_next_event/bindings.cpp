#include "get_next_event.hpp"
#include "get_next_event.cpp"
#include <pybind11/pybind11.h>
#include<pybind11/stl.h>

namespace py = pybind11;


PYBIND11_MODULE(c_imp_get_next_event, m){
    m.doc() = "test";
    m.def("get_next_event", &get_next_event, "Gets the next event");

    py::class_<next_event>(m, "next_event")
    .def(py::init<>())
    .def_readwrite("shortest_distance_to_next_event", &next_event::shortest_distance_to_next_event)
    .def_readwrite("vetoing_index", &next_event::vetoing_index);



}