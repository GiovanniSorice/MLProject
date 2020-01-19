cmake_minimum_required(VERSION 3.15)
project(MLProject)

if(EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup()
else()
    message(WARNING "The file conanbuildinfo.cmake doesn't exist, you have to run conan install first")
endif()

set(CMAKE_CXX_STANDARD 17)


add_executable(MLProject main.cpp src/layer/layer.cpp src/layer/layer.h src/network/network.cpp src/network/network.h src/preprocessing/preprocessing.cpp src/preprocessing/preprocessing.h src/preprocessing/preprocessing.cpp src/preprocessing/preprocessing.h src/activationFunction/activationFunction.h src/activationFunction/tanhFunction.cpp src/activationFunction/tanhFunction.h src/activationFunction/logisticFunction.cpp src/activationFunction/logisticFunction.h src/load/loadDataset.cpp src/load/loadDataset.h src/lossFunction/lossFunction.h src/lossFunction/meanSquaredError.cpp src/lossFunction/meanSquaredError.h src/activationFunction/reluFunction.cpp src/activationFunction/reluFunction.h src/lossFunction/binaryCrossentropy.cpp src/lossFunction/binaryCrossentropy.h src/gridSearch/gridSearch.cpp src/gridSearch/gridSearch.h)
target_link_libraries(MLProject ${CONAN_LIBS})