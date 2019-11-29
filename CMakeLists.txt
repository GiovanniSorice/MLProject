cmake_minimum_required(VERSION 3.15)
project(MLProject)

if(EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup()
else()
    message(WARNING "The file conanbuildinfo.cmake doesn't exist, you have to run conan install first")
endif()

set(CMAKE_CXX_STANDARD 17)

add_executable(MLProject main.cpp src/layer/layer.cpp src/layer/layer.h src/network/network.cpp src/network/network.h)