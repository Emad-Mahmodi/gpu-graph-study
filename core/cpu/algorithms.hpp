// core/cpu/algorithms.hpp
// Tier T0 — sequential C++ reference implementations.
// These are the CORRECTNESS ORACLE: every faster tier (OpenMP, CUDA) is
// validated bit-for-bit against these results.
#pragma once

#include <cstdint>
#include <vector>

#include "../graph/csr.hpp"

namespace ggs::seq {

// --- BFS -----------------------------------------------------------------
// Returns the distance (in hops) from `source` to every vertex.
// Unreached vertices get UNREACHED.
constexpr std::int32_t UNREACHED = -1;
std::vector<std::int32_t> bfs(const CsrGraph& g, VertexId source);

// --- PageRank ------------------------------------------------------------
// Standard PageRank with damping factor `damping`. Runs up to `max_iters`
// or until the L1 change drops below `tol`. Dangling mass is redistributed.
std::vector<double> pagerank(const CsrGraph& g,
                             double damping   = 0.85,
                             int    max_iters = 100,
                             double tol       = 1e-9);

// --- Weakly Connected Components -----------------------------------------
// Label-propagation / BFS flood. Returns a component id per vertex
// (the smallest vertex id in each component, so labels are canonical).
std::vector<VertexId> wcc(const CsrGraph& g);

// Number of distinct components given a wcc() labeling.
VertexId count_components(const std::vector<VertexId>& labels);

}  // namespace ggs::seq
