#include "cpp_quantum_harmonic_oscillator.hpp"
#include "cpp_quantum_harmonic_oscillator.cpp"
#include <pybind11/pybind11.h>
#include<pybind11/stl.h>

namespace py = pybind11;


PYBIND11_MODULE(cpp_qho, m){
    m.doc() = "test";
    m.def("get_next_event", &get_next_event, "Gets the next event");
    m.def("get_potential_difference", &get_potential_difference, "Gets the potential difference between the candidate and current actions");

    py::class_<next_event>(m, "next_event")
    .def(py::init<>())
    .def_readwrite("shortest_distance_to_next_event", &next_event::shortest_distance_to_next_event)
    .def_readwrite("vetoing_index", &next_event::vetoing_index);



}