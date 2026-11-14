
cmake_minimum_required(VERSION 3.15)
project(cpp_binpack_loader LANGUAGES CXX)
find_package(Python3 REQUIRED COMPONENTS Development)
find_package(pybind11 REQUIRED)
pybind11_add_module(cpp_loader src/binpack_loader.cpp src/bindings.cpp)
set_target_properties(cpp_loader PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    PYTHON_MODULE_SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
set_target_properties(cpp_loader PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..
)
