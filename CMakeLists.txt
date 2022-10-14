cmake_minimum_required(VERSION 3.4...3.18)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_C_COMPILER "gcc")

project(dessert_py)

project(dessert_py LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(
    -Wall
    -Wextra
    # TODO(Josh): Fix eigen error and add back -werror
    -Wno-unused-function
    -Wno-unused-parameter
    -pedantic 
    -Ofast
    -g
    -fno-omit-frame-pointer
)

find_package (Python3 COMPONENTS Interpreter Development)

# Header only dependencies
include_directories(deps/pybind/include)
include_directories(deps/eigen)
include_directories(deps/cereal/include)

# Source dependencies
add_subdirectory(deps/pybind)

# So we can include using from the base directory
include_directories(".")

pybind11_add_module(dessert_py src/MaxFlash.cc src/MaxFlashArray.cc src/SRP.cc python_bindings/DessertPython.cc)

find_package(OpenMP REQUIRED)

target_link_libraries(dessert_py PUBLIC OpenMP::OpenMP_CXX)
