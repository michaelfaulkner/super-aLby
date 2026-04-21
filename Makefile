PYTHON ?= python
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
CXX ?= c++
CXXFLAGS := -O3 -Wall -shared -std=c++11 -fPIC $(PYBIND11_INCLUDES)

SRC := src/potential/cpp_next_event/bindings.cpp
TARGET := src/potential/cpp_next_event/c_imp_get_next_event.so

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean