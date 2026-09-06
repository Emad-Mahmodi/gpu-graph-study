// core/cpu/louvain_omp.hpp
// Tier T1 — OpenMP parallel Louvain community detection.
//
// Unlike BFS / PageRank / WCC, this tier cannot return the *same* answer as the
// T0 oracle, and pretending otherwise would be dishonest. Sequential Louvain is
// Gauss-Seidel: it moves one vertex at a time and every move is immediately
// visible to the next vertex, so the outcome depends on the visiting order. A
// parallel version has to give that up. This one is Jacobi — every vertex picks
// its best move from the *same* snapshot of the assignment, and all moves are
// applied at a barrier — which is the standard parallel formulation (Lu,
// Halappanavar & Kalyanaraman, "Parallel heuristics for scalable community
// detection", Parallel Computing 2015).
//
// Two consequences, both measured in the tests rather than assumed:
//
//   * The communities differ from T0's. The contract is therefore on *quality*
//     (modularity within a small tolerance of T0), not on label equality.
//   * The result is nonetheless **deterministic and independent of the thread
//     count**: a vertex's choice depends only on the snapshot and on the fixed
//     edge order, so 1 thread and 16 threads produce identical communities.
//     Sequential Louvain has no such guarantee across orderings. This is a
//     genuine advantage of the parallel formulation, not a workaround.
#pragma once

#include "../graph/csr.hpp"
#include "louvain.hpp"   // reuse seq::LouvainResult

namespace ggs::omp {

// Same signature and result type as ggs::seq::louvain.
seq::LouvainResult louvain(const CsrGraph& g, int max_passes = 100,
                           double min_gain = 1e-9);

}  // namespace ggs::omp
