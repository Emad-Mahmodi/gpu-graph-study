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
#include "../cpu/louvain.hpp"   // reuse seq::LouvainResult

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

// Tier T4 — degree-bucketed BFS. Unlike PageRank, whose gather touches every
// vertex every iteration, BFS only expands the current frontier — so the
// bucketing cannot be done once on the host. Each level the frontier is
// re-partitioned *on the device* by out-degree into three sub-frontiers, and
// each is expanded at the granularity that fits:
//   low  (<= 32)     one thread per vertex
//   mid  (33..1024)  one warp per vertex  (32 lanes stride col[], coalesced)
//   high (> 1024)    one block per vertex (256 threads on a single hub)
// This is the roadmap's "re-bucket between BFS levels". Distances are identical
// to the T0 oracle.
std::vector<std::int32_t> bfs_lb(const CsrGraph& g, VertexId source);

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

// Tier T4 — degree-bucketed load balancing (established Gunrock/Merrill
// technique, not a novel contribution). Vertices are split once by out-degree:
// low-degree vertices keep one thread each, high-degree hubs get a whole warp
// (32 lanes stride the neighbor list, coalesced, then a warp-shuffle reduction).
// This attacks the power-law load imbalance that a single thread on a 10^5-degree
// hub causes. Same result as the T0 oracle.
std::vector<double> pagerank_lb(const CsrGraph& g,
                                double damping   = 0.85,
                                int    max_iters = 100,
                                double tol       = 1e-9);

std::vector<VertexId> wcc(const CsrGraph& g);

// Tier T3 — optimized CUDA WCC: active-set label propagation. Only vertices
// whose neighborhood changed in the previous round are reprocessed, instead of
// sweeping all N vertices every round (T2). Same component labels as the T0
// oracle.
std::vector<VertexId> wcc_opt(const CsrGraph& g);

// Tier T4 — degree-bucketed WCC. Keeps the T3 active-set (only vertices woken
// last round are rescanned) but fixes the *other* half of the problem: a hub
// with 10^5 neighbours was still scanned by a single thread. Degrees are static
// here, so the three buckets are built once on the host, as in PageRank; each
// round launches thread-/warp-/block-per-vertex kernels over them. The
// warp/block kernels min-reduce the neighbour labels with __shfl_down_sync /
// a shared-memory tree. Labels are identical to the T0 oracle.
std::vector<VertexId> wcc_lb(const CsrGraph& g);

// Tier T2 - naive CUDA Louvain community detection.
//
// Louvain needs something the other three algorithms never do: for each vertex,
// the sum of edge weights to *each distinct neighbouring community*. On the CPU
// that is a dense array of size n per thread. With tens of thousands of GPU
// threads in flight that is not an option, and it is the whole difficulty of
// putting Louvain on a GPU.
//
// The obvious answer is a per-warp hash table in shared memory -- but a hub with
// half a million neighbours can touch more distinct communities than any shared
// table holds, so a hash table needs an overflow path, which is most of the work
// the T4 tier is supposed to isolate. Doing it here would make the later
// speedup unattributable.
//
// So T2 does the plainly correct thing instead: label every arc with
// (owner vertex, neighbour community), sort, and segment-reduce. No capacity
// limit, no special case, no tuning -- and predictably slow, which is exactly
// what an honest baseline tier is for.
//
// Like the OpenMP tier, this cannot reproduce the T0 oracle's labels (sequential
// Louvain is order-dependent by construction), so it is contracted on modularity
// rather than on equality.
seq::LouvainResult louvain(const CsrGraph& g, int max_passes = 100,
                           double min_gain = 1e-9);

}  // namespace ggs::cuda
