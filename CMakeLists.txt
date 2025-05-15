cmake_minimum_required(VERSION 3.18)
project(cyclic_product_graphs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

## load external content
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# Libigl
include(libigl)
include(robin-map)
include(thrust)
thrust_create_target(Thrust)

find_package(OpenMP)

if(WITH_CUDA)
    message(STATUS "Compiling with CUDA support")
    add_definitions(-DWITH_CUDA)
    enable_language(CUDA)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    find_package(CUDAToolkit)
    #set(CMAKE_CUDA_ARCHITECTURES 75)
endif()


# switch between python bindings and executable
if( BUILD_PYTHON_BINDINGS )
    # pybind11
    include(pybind)
    message("Building Python Bindings")
    find_package(Python3 REQUIRED)
    include_directories(${PYTHON_INCLUDE_DIRS})
    pybind11_add_module(cyclic_product_graphs pythonbinding/bindings.cpp)
    add_subdirectory(pythonbinding)
    target_compile_options(cyclic_product_graphs PRIVATE -w)
elseif( BUILD_MEX_FILE )
    message("Building MEX files")
    find_package(Matlab REQUIRED)
    set(PROJECT_NAME product_graph_generator_mex)
    add_subdirectory(mex)
else()
    message("Building CPP Executable")
    add_executable(${PROJECT_NAME} main.cpp)
endif()

# set project include dir
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR})


# Add your project files
add_subdirectory(helper)
add_subdirectory(product_graph_generators)


if( NOT BUILD_MEX_FILE )
    if(OpenMP_CXX_FOUND)
        message("Compiling with OPENMP")
        if(WITH_CUDA)
            target_link_libraries(${PROJECT_NAME} PUBLIC igl::core tsl::robin_map Thrust OpenMP::OpenMP_CXX CUDA::cudart)
        else()
            target_link_libraries(${PROJECT_NAME} PUBLIC igl::core tsl::robin_map Thrust OpenMP::OpenMP_CXX)
        endif()
    else()
        message("Compiling without OPENMP")
        if(WITH_CUDA)
            target_link_libraries(${PROJECT_NAME} PUBLIC igl::core tsl::robin_map Thrust CUDA::cudart)
        else()
            target_link_libraries(${PROJECT_NAME} PUBLIC igl::core tsl::robin_map Thrust)
        endif()
    endif()
endif()

if(OpenMP_CXX_FOUND AND WITH_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp -Xlinker -fgomp")
endif()