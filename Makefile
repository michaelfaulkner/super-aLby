PYTHON ?= python
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
CXX ?= c++
CXXFLAGS := -O3 -Wall -shared -std=c++11 -fPIC $(PYBIND11_INCLUDES)

SRC := src/potential/cpp_next_event/bindings.cpp
TARGET := src/potential/cpp_next_event/c_imp_get_next_event.so

SRC_PD := src/potential/cpp_potential_difference/bindings.cpp
TARGET_PD := src/potential/cpp_potential_difference/c_imp_get_potential_difference.so

all: $(TARGET) $(TARGET_PD)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

$(TARGET_PD): $(SRC_PD)
	$(CXX) $(CXXFLAGS) $(SRC_PD) -o $(TARGET_PD)

clean:
	rm -f $(TARGET) $(TARGET_PD)

.PHONY: all clean