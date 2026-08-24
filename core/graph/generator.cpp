// core/graph/generator.cpp
#include "generator.hpp"

#include <cstdint>
#include <random>

namespace ggs {

CsrGraph make_rmat(int scale, int edge_factor, std::uint64_t seed,
                   bool symmetrize, double a, double b, double c) {
    const VertexId n = static_cast<VertexId>(1) << scale;
    const EdgeId   m = static_cast<EdgeId>(edge_factor) * n;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::vector<std::pair<VertexId, VertexId>> edges;
    edges.reserve(m);

    const double ab  = a + b;
    const double abc = a + b + c;

    for (EdgeId e = 0; e < m; ++e) {
        VertexId u = 0, v = 0;
        // Recursively descend the adjacency-matrix quadrants.
        for (int bit = scale - 1; bit >= 0; --bit) {
            double r = uni(rng);
            if (r < a) {
                // top-left: (0,0)
            } else if (r < ab) {
                v |= (static_cast<VertexId>(1) << bit);          // top-right
            } else if (r < abc) {
                u |= (static_cast<VertexId>(1) << bit);          // bottom-left
            } else {
                u |= (static_cast<VertexId>(1) << bit);          // bottom-right
                v |= (static_cast<VertexId>(1) << bit);
            }
        }
        if (u == v) continue;   // drop self-loops
        edges.emplace_back(u, v);
    }
    return CsrGraph::from_edges(n, edges, symmetrize);
}

CsrGraph make_test_graph() {
    // Component A (0,1,2,3): edges 0-1, 1-2, 2-3, 0-3
    // Component B (4,5,6):   edges 4-5, 5-6, 4-6
    std::vector<std::pair<VertexId, VertexId>> edges = {
        {0, 1}, {1, 2}, {2, 3}, {0, 3},
        {4, 5}, {5, 6}, {4, 6},
    };
    return CsrGraph::from_edges(7, edges, /*symmetrize=*/true);
}

}  // namespace ggs
