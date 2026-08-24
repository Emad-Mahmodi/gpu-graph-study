// core/cpu/algorithms.cpp  — Tier T0 sequential reference.
#include "algorithms.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>

namespace ggs::seq {

std::vector<std::int32_t> bfs(const CsrGraph& g, VertexId source) {
    std::vector<std::int32_t> dist(g.num_vertices, UNREACHED);
    if (g.num_vertices == 0) return dist;

    std::queue<VertexId> frontier;
    dist[source] = 0;
    frontier.push(source);

    while (!frontier.empty()) {
        VertexId u = frontier.front();
        frontier.pop();
        for (EdgeId e = g.row_offsets[u]; e < g.row_offsets[u + 1]; ++e) {
            VertexId v = g.col_indices[e];
            if (dist[v] == UNREACHED) {
                dist[v] = dist[u] + 1;
                frontier.push(v);
            }
        }
    }
    return dist;
}

std::vector<double> pagerank(const CsrGraph& g, double damping,
                             int max_iters, double tol) {
    const VertexId n = g.num_vertices;
    std::vector<double> rank(n, n ? 1.0 / n : 0.0);
    std::vector<double> next(n, 0.0);
    if (n == 0) return rank;

    const double base = (1.0 - damping) / n;

    for (int it = 0; it < max_iters; ++it) {
        // Collect dangling (zero out-degree) mass to redistribute.
        double dangling = 0.0;
        for (VertexId v = 0; v < n; ++v)
            if (g.degree(v) == 0) dangling += rank[v];
        const double dangling_share = damping * dangling / n;

        std::fill(next.begin(), next.end(), base + dangling_share);

        // Push each vertex's rank to its out-neighbors.
        for (VertexId u = 0; u < n; ++u) {
            EdgeId d = g.degree(u);
            if (d == 0) continue;
            double contrib = damping * rank[u] / static_cast<double>(d);
            for (EdgeId e = g.row_offsets[u]; e < g.row_offsets[u + 1]; ++e)
                next[g.col_indices[e]] += contrib;
        }

        double diff = 0.0;
        for (VertexId v = 0; v < n; ++v) diff += std::fabs(next[v] - rank[v]);
        rank.swap(next);
        if (diff < tol) break;
    }
    return rank;
}

std::vector<VertexId> wcc(const CsrGraph& g) {
    const VertexId n = g.num_vertices;
    std::vector<VertexId> label(n, UNREACHED);  // UNREACHED reused as "unlabeled"

    for (VertexId s = 0; s < n; ++s) {
        if (label[s] != static_cast<VertexId>(UNREACHED)) continue;
        // Flood from s; canonical label is s (the smallest unvisited id
        // encountered starts each new component, so labels stay minimal).
        label[s] = s;
        std::queue<VertexId> q;
        q.push(s);
        while (!q.empty()) {
            VertexId u = q.front();
            q.pop();
            for (EdgeId e = g.row_offsets[u]; e < g.row_offsets[u + 1]; ++e) {
                VertexId v = g.col_indices[e];
                if (label[v] == static_cast<VertexId>(UNREACHED)) {
                    label[v] = s;
                    q.push(v);
                }
            }
        }
    }
    return label;
}

VertexId count_components(const std::vector<VertexId>& labels) {
    VertexId count = 0;
    for (VertexId v = 0; v < labels.size(); ++v)
        if (labels[v] == v) ++count;   // canonical label == own id => a root
    return count;
}

}  // namespace ggs::seq
