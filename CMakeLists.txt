cmake_minimum_required(VERSION 3.18)
project(gpu_graph_study LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Optimization flags differ by compiler: MSVC uses /O2 (its Release default),
# GCC/Clang use -O3.
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

# --- Core engine library (Tier T0 today; OpenMP/CUDA tiers added later) ----
add_library(ggs_core
  core/graph/csr.cpp
  core/graph/generator.cpp
  core/cpu/algorithms.cpp
  core/cpu/algorithms_omp.cpp
  core/cpu/louvain.cpp
)
target_include_directories(ggs_core PUBLIC ${CMAKE_SOURCE_DIR})

# Optional OpenMP — wired now so Phase 2 (T1) drops in without CMake changes.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(ggs_core PUBLIC OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP found — T1 tier can be enabled.")
endif()

# --- Optional CUDA tier (T2+) ---------------------------------------------
# Built only if a CUDA compiler is present, so the project still builds
# CPU-only (CI, machines without an NVIDIA GPU). Defines HAVE_CUDA downstream.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  add_library(ggs_cuda STATIC core/cuda/cuda_algorithms.cu)
  target_include_directories(ggs_cuda PUBLIC ${CMAKE_SOURCE_DIR})
  # Turing (T600) = compute capability 7.5. Add more archs if you retarget.
  set_target_properties(ggs_cuda PROPERTIES CUDA_ARCHITECTURES 75)
  message(STATUS "CUDA found — T2 GPU tier enabled (arch sm_75).")
endif()

# --- Benchmark driver ------------------------------------------------------
add_executable(run_bench apps/run_bench.cpp)
target_link_libraries(run_bench PRIVATE ggs_core)
if(CMAKE_CUDA_COMPILER)
  target_link_libraries(run_bench PRIVATE ggs_cuda)
  target_compile_definitions(run_bench PRIVATE HAVE_CUDA)
endif()

# --- Tests -----------------------------------------------------------------
enable_testing()
add_executable(test_correctness tests/test_correctness.cpp)
target_link_libraries(test_correctness PRIVATE ggs_core)
# When CUDA is present, the test also validates the T2/T3 GPU kernels against
# the T0 oracle (guarded by HAVE_CUDA inside the source).
if(CMAKE_CUDA_COMPILER)
  target_link_libraries(test_correctness PRIVATE ggs_cuda)
  target_compile_definitions(test_correctness PRIVATE HAVE_CUDA)
endif()
add_test(NAME correctness COMMAND test_correctness)

# Placeholder for the future CUDA target:
#   enable_language(CUDA)  # Phase 3
