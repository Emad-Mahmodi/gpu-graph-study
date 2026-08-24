// core/graph/csr.cpp
#include "csr.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ggs {

CsrGraph CsrGraph::from_edges(
        VertexId num_vertices,
        const std::vector<std::pair<VertexId, VertexId>>& edges,
        bool symmetrize) {
    CsrGraph g;
    g.num_vertices = num_vertices;

    // Count degrees first (counting sort into CSR).
    std::vector<EdgeId> deg(num_vertices + 1, 0);
    for (const auto& [u, v] : edges) {
        deg[u]++;
        if (symmetrize) deg[v]++;
    }

    // Prefix sum -> row_offsets.
    g.row_offsets.assign(num_vertices + 1, 0);
    for (VertexId i = 0; i < num_vertices; ++i)
        g.row_offsets[i + 1] = g.row_offsets[i] + deg[i];
    g.num_edges = g.row_offsets[num_vertices];

    // Scatter neighbors using a moving cursor per row.
    g.col_indices.resize(g.num_edges);
    std::vector<EdgeId> cursor(g.row_offsets.begin(), g.row_offsets.end());
    for (const auto& [u, v] : edges) {
        g.col_indices[cursor[u]++] = v;
        if (symmetrize) g.col_indices[cursor[v]++] = u;
    }
    return g;
}

CsrGraph CsrGraph::load_edgelist(const std::string& path, bool symmetrize) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open graph file: " + path);

    std::vector<std::pair<VertexId, VertexId>> edges;
    VertexId max_id = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        VertexId u, v;
        if (!(ss >> u >> v)) continue;   // skip malformed lines
        edges.emplace_back(u, v);
        max_id = std::max(max_id, std::max(u, v));
    }
    return from_edges(max_id + 1, edges, symmetrize);
}

CsrGraph::Stats CsrGraph::stats() const {
    Stats s{num_vertices, num_edges, 0, 0.0, 0};
    for (VertexId v = 0; v < num_vertices; ++v) {
        EdgeId d = degree(v);
        s.max_degree = std::max(s.max_degree, d);
        if (d == 0) s.num_zero_degree++;
    }
    s.avg_degree = num_vertices ? static_cast<double>(num_edges) / num_vertices : 0.0;
    return s;
}

}  // namespace ggs
