cmake_minimum_required(VERSION 3.18)
project(lbm_solver CUDA CXX)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 61)  # Change if needed for your GPU

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Make sure we have the cudadevrt library available
find_library(CUDA_cudadevrt_LIBRARY cudadevrt
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64
    NO_DEFAULT_PATH
)

# Add include directories
include_directories(src)

# Ensure CUDA files are properly identified by CMake
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

# Find all source files, separating CUDA and C++ files
file(GLOB_RECURSE CXX_SOURCES 
    "src/*.cpp"
    "src/*.hpp"
)

file(GLOB_RECURSE CUDA_SOURCES 
    "src/*.cu"
    "src/*.cuh"
)

# Add executable with explicit CUDA and C++ sources
add_executable(lbm_solver ${CUDA_SOURCES} ${CXX_SOURCES})

# Enable CUDA separable compilation
set_target_properties(lbm_solver PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/windows"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/windows"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/windows"
)

# Link CUDA libraries
target_link_libraries(lbm_solver ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY} cudadevrt)

# Compiler options - fixed to avoid conflicts between optimization and debug flags
# Different settings for Debug and Release modes
set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")

# Release mode settings
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -rdc=true")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

# Debug mode settings - removed optimization flags
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -rdc=true")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

# Use default library for MSVC to avoid warning LNK4098
if(MSVC)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
endif()

# Print some configuration information
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "CUDA Debug Flags: ${CMAKE_CUDA_FLAGS_DEBUG}")
message(STATUS "CUDA Release Flags: ${CMAKE_CUDA_FLAGS_RELEASE}")
message(STATUS "CUDA Sources: ${CUDA_SOURCES}")