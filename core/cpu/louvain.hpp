// core/cpu/louvain.hpp
// Tier T0 — sequential Louvain community detection.
// Multi-level modularity optimization on the (undirected, symmetrized) graph.
// The 4th algorithm in the study; the hardest to parallelize, so it lives at T0
// first. Returns a community label per vertex and the final modularity Q.
#pragma once

#include <cstdint>
#include <vector>

#include "../graph/csr.hpp"

namespace ggs::seq {

struct LouvainResult {
    std::vector<VertexId> community;  // community id per original vertex
    double modularity = 0.0;          // final modularity Q in [-0.5, 1]
    int    levels = 0;                // number of aggregation levels performed
    VertexId num_communities = 0;
};

// Run Louvain. `max_passes` bounds the local-moving sweeps per level.
// The graph is treated as undirected with unit edge weights (our CSR is
// symmetrized), which is the standard Louvain setting.
LouvainResult louvain(const CsrGraph& g, int max_passes = 100,
                      double min_gain = 1e-9);

}  // namespace ggs::seq
