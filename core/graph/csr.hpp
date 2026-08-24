// core/graph/csr.hpp
// Compressed Sparse Row (CSR) graph representation.
// This is the ONE data structure every algorithm tier consumes.
// The analytics engine never sees a binary or a packet — only this.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ggs {

using VertexId = std::uint32_t;
using EdgeId   = std::uint64_t;

// A directed graph in CSR form.
//   row_offsets : size num_vertices + 1
//   col_indices : size num_edges
// The out-neighbors of vertex v are
//   col_indices[row_offsets[v] .. row_offsets[v+1]-1]
struct CsrGraph {
    VertexId              num_vertices = 0;
    EdgeId                num_edges    = 0;
    std::vector<EdgeId>   row_offsets;   // CSR pointers
    std::vector<VertexId> col_indices;   // CSR neighbors

    // degree(v) = number of out-neighbors of v
    inline EdgeId degree(VertexId v) const {
        return row_offsets[v + 1] - row_offsets[v];
    }

    // --- Construction -----------------------------------------------------

    // Build a CSR graph from an edge list.
    // If `symmetrize` is true, every edge (u,v) also inserts (v,u) — use this
    // for undirected algorithms (WCC, PageRank on undirected, Louvain).
    static CsrGraph from_edges(VertexId num_vertices,
                               const std::vector<std::pair<VertexId, VertexId>>& edges,
                               bool symmetrize = false);

    // --- I/O --------------------------------------------------------------

    // Load an edge-list text file. Format: one edge per line "u v",
    // lines starting with '#' are comments, blank lines ignored.
    // num_vertices is inferred as (max id + 1) unless given explicitly.
    static CsrGraph load_edgelist(const std::string& path,
                                  bool symmetrize = false);

    // Basic structural stats used in the empirical study tables.
    struct Stats {
        VertexId num_vertices;
        EdgeId   num_edges;
        EdgeId   max_degree;
        double   avg_degree;
        VertexId num_zero_degree;
    };
    Stats stats() const;
};

}  // namespace ggs
