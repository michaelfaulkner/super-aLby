cmake_minimum_required(VERSION 3.4...3.18)
project(get_next_event)

add_library(next_event_code STATIC src/cpp_next_event/get_next_event.cpp)
set_target_properties(next_event_code PROPERTIES PPOSITION_INDEPENDENT_CODE ON)

add_subdirectory(pybind11)
pybind11_add_module(get_next_event src/cpp_next_event/bindings.cpp)
target_link_libraries(get_next_event PRIVATE next_event_code)