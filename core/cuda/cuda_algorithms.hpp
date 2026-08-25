// core/cuda/cuda_algorithms.hpp
// Tier T2 — naïve CUDA implementations. Host-callable declarations only (no CUDA
// syntax here) so plain C++ translation units (run_bench, tests) can include it.
// Same signatures as the sequential (T0) oracle; every result is validated
// against T0. "Naïve" = one thread per vertex, CSR in global memory, no shared
// memory or warp tricks yet — the honest baseline the optimized tiers improve on.
#pragma once

#include <cstdint>
#include <vector>

#include "../graph/csr.hpp"

namespace ggs::cuda {

// True if at least one CUDA device is visible at runtime.
bool available();

// Human-readable name of device 0 (or "" if none).
std::string device_name();

std::vector<std::int32_t> bfs(const CsrGraph& g, VertexId source);

// Tier T3 — optimized CUDA BFS: an explicit frontier worklist. Instead of
// scanning all N vertices at every level (T2), only the current frontier is
// launched; newly discovered vertices are claimed with atomicCAS and appended
// to the next frontier. Same distances as the T0 oracle.
std::vector<std::int32_t> bfs_opt(const CsrGraph& g, VertexId source);

std::vector<double> pagerank(const CsrGraph& g,
                             double damping   = 0.85,
                             int    max_iters = 100,
                             double tol       = 1e-9);

// Tier T3 — optimized CUDA PageRank. Same result as the naïve (T2) kernel
// (validated against the T0 oracle), but the two per-iteration reductions
// (dangling mass, L1 change) are computed with block-level parallel reductions
// in shared memory — one atomicAdd per block instead of one per vertex — and
// the inverse out-degree is precomputed once instead of divided every
// iteration. This is what turns PageRank from a GPU loss into a win.
std::vector<double> pagerank_opt(const CsrGraph& g,
                                 double damping   = 0.85,
                                 int    max_iters = 100,
                                 double tol       = 1e-9);

std::vector<VertexId> wcc(const CsrGraph& g);

// Tier T3 — optimized CUDA WCC: active-set label propagation. Only vertices
// whose neighborhood changed in the previous round are reprocessed, instead of
// sweeping all N vertices every round (T2). Same component labels as the T0
// oracle.
std::vector<VertexId> wcc_opt(const CsrGraph& g);

}  // namespace ggs::cuda
