cmake_minimum_required(VERSION 3.18)
project(lbm_solver CUDA CXX)

set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS
    .cu;.cuh;.cpp;.cxx;.cc
)

set(CMAKE_CUDA_ARCHITECTURES 61)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(src)

include_directories(third_party)

file(GLOB_RECURSE CXX_SOURCES "src/*.cpp" "src/*.hpp")
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu" "src/*.cuh")

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

add_executable(lbm_solver ${CUDA_SOURCES} ${CXX_SOURCES})

find_package(CUDAToolkit REQUIRED)
target_link_libraries(lbm_solver
  PRIVATE
    CUDA::cudart_static
)

set_target_properties(lbm_solver PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Static
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/windows"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/windows"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/windows"
)

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -rdc=true")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -rdc=true")

if(MSVC)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
endif()

message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")