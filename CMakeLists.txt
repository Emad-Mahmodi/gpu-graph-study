cmake_minimum_required(VERSION 3.18)
project(gpu_graph_study LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# --- Core engine library (Tier T0 today; OpenMP/CUDA tiers added later) ----
add_library(ggs_core
  core/graph/csr.cpp
  core/graph/generator.cpp
  core/cpu/algorithms.cpp
)
target_include_directories(ggs_core PUBLIC ${CMAKE_SOURCE_DIR})

# Optional OpenMP — wired now so Phase 2 (T1) drops in without CMake changes.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(ggs_core PUBLIC OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP found — T1 tier can be enabled.")
endif()

# --- Benchmark driver ------------------------------------------------------
add_executable(run_bench apps/run_bench.cpp)
target_link_libraries(run_bench PRIVATE ggs_core)

# --- Tests -----------------------------------------------------------------
enable_testing()
add_executable(test_correctness tests/test_correctness.cpp)
target_link_libraries(test_correctness PRIVATE ggs_core)
add_test(NAME correctness COMMAND test_correctness)

# Placeholder for the future CUDA target:
#   enable_language(CUDA)  # Phase 3
