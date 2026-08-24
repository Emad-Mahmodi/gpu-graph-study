// core/cpu/algorithms_omp.hpp
// Tier T1 — OpenMP multi-core CPU implementations.
// Same signatures as the sequential (T0) oracle in algorithms.hpp; every T1
// result is diffed against T0 in the tests. This is the baseline that GPU
// speedups (T2+) are reported against — never against a single core.
#pragma once

#include <cstdint>
#include <vector>

#include "../graph/csr.hpp"
#include "algorithms.hpp"   // reuse UNREACHED and result conventions

namespace ggs::omp {

// Level-synchronous, frontier-based parallel BFS.
std::vector<std::int32_t> bfs(const CsrGraph& g, VertexId source);

// Push-based parallel PageRank (atomic accumulation).
std::vector<double> pagerank(const CsrGraph& g,
                             double damping   = 0.85,
                             int    max_iters = 100,
                             double tol       = 1e-9);

// Parallel label-propagation weakly-connected components.
std::vector<VertexId> wcc(const CsrGraph& g);

}  // namespace ggs::omp
