// core/cpu/algorithms_omp.cpp  — Tier T1 (OpenMP).
#include "algorithms_omp.hpp"

#include <atomic>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ggs::omp {

// ---------------------------------------------------------------------------
// BFS — level-synchronous. Each level expands the whole current frontier in
// parallel; a vertex is claimed exactly once via an atomic compare-exchange on
// its distance. Result is identical to sequential BFS (same distances).
// ---------------------------------------------------------------------------
std::vector<std::int32_t> bfs(const CsrGraph& g, VertexId source) {
    const VertexId n = g.num_vertices;
    // std::atomic gives a lock-free claim; -1 == UNREACHED.
    std::vector<std::atomic<std::int32_t>> dist(n);
    for (VertexId v = 0; v < n; ++v)
        dist[v].store(seq::UNREACHED, std::memory_order_relaxed);
    std::vector<std::int32_t> out(n, seq::UNREACHED);
    if (n == 0) return out;

    dist[source].store(0, std::memory_order_relaxed);
    std::vector<VertexId> frontier{source};

    std::int32_t level = 0;
    while (!frontier.empty()) {
        std::vector<VertexId> next;
        const long long fsize = static_cast<long long>(frontier.size());
        // Per-thread buffers avoid contention on a shared next-frontier;
        // merged once per level. Signed loop index required by MSVC OpenMP.
        #pragma omp parallel
        {
            std::vector<VertexId> local;
            #pragma omp for nowait schedule(dynamic, 256)
            for (long long i = 0; i < fsize; ++i) {
                VertexId u = frontier[static_cast<std::size_t>(i)];
                for (EdgeId e = g.row_offsets[u]; e < g.row_offsets[u + 1]; ++e) {
                    VertexId v = g.col_indices[e];
                    std::int32_t expected = seq::UNREACHED;
                    // Claim v for this level if still unreached.
                    if (dist[v].compare_exchange_strong(
                            expected, level + 1, std::memory_order_relaxed)) {
                        local.push_back(v);
                    }
                }
            }
            #pragma omp critical
            next.insert(next.end(), local.begin(), local.end());
        }
        frontier.swap(next);
        ++level;
    }

    for (VertexId v = 0; v < n; ++v)
        out[v] = dist[v].load(std::memory_order_relaxed);
    return out;
}

// ---------------------------------------------------------------------------
// PageRank — PULL-based. Each vertex v gathers damping*rank[u]/deg(u) from its
// neighbors u. This avoids the atomic scatter that cripples a push kernel on
// power-law graphs (every thread hammering the same hub slot), so it scales
// cleanly. Exact for undirected / symmetrized graphs — the benchmark setting
// here — where a vertex's CSR neighbor list is also its in-neighbor list.
// (A directed pull would gather over the transpose/CSC instead.)
// ---------------------------------------------------------------------------
std::vector<double> pagerank(const CsrGraph& g, double damping,
                             int max_iters, double tol) {
    const VertexId n = g.num_vertices;
    std::vector<double> rank(n, n ? 1.0 / n : 0.0);
    std::vector<double> next(n, 0.0);
    if (n == 0) return rank;

    const double base = (1.0 - damping) / n;
    const long long nn = static_cast<long long>(n);  // signed for MSVC OpenMP

    for (int it = 0; it < max_iters; ++it) {
        // Dangling mass (zero out-degree) redistributed uniformly.
        double dangling = 0.0;
        #pragma omp parallel for reduction(+:dangling) schedule(static)
        for (long long v = 0; v < nn; ++v)
            if (g.degree(static_cast<VertexId>(v)) == 0)
                dangling += rank[static_cast<VertexId>(v)];
        const double add = base + damping * dangling / n;

        // Each vertex computes its own next value independently — no atomics.
        double diff = 0.0;
        #pragma omp parallel for reduction(+:diff) schedule(dynamic, 256)
        for (long long vv = 0; vv < nn; ++vv) {
            VertexId v = static_cast<VertexId>(vv);
            double sum = 0.0;
            for (EdgeId e = g.row_offsets[v]; e < g.row_offsets[v + 1]; ++e) {
                VertexId u = g.col_indices[e];
                sum += rank[u] / static_cast<double>(g.degree(u));
            }
            double nv = add + damping * sum;
            next[v] = nv;
            diff += std::fabs(nv - rank[v]);
        }
        rank.swap(next);
        if (diff < tol) break;
    }
    return rank;
}

// ---------------------------------------------------------------------------
// WCC — parallel label propagation. label[v] starts at v; each round every
// vertex lowers its label to the min over its neighbors. Monotonic (labels
// only decrease), so races are benign and the fixed point is the min vertex id
// per component — identical component structure to sequential BFS-flood.
// ---------------------------------------------------------------------------
std::vector<VertexId> wcc(const CsrGraph& g) {
    const VertexId n = g.num_vertices;
    const long long nn = static_cast<long long>(n);  // signed for MSVC OpenMP
    std::vector<VertexId> label(n);
    #pragma omp parallel for schedule(static)
    for (long long v = 0; v < nn; ++v)
        label[static_cast<VertexId>(v)] = static_cast<VertexId>(v);

    // int (not bool) for OpenMP 2.0 reduction compatibility on MSVC.
    int changed = 1;
    while (changed) {
        changed = 0;
        #pragma omp parallel for schedule(dynamic, 256) reduction(|:changed)
        for (long long uu = 0; uu < nn; ++uu) {
            VertexId u = static_cast<VertexId>(uu);
            VertexId best = label[u];
            for (EdgeId e = g.row_offsets[u]; e < g.row_offsets[u + 1]; ++e) {
                VertexId lv = label[g.col_indices[e]];
                if (lv < best) best = lv;
            }
            if (best < label[u]) { label[u] = best; changed = 1; }
        }
    }
    return label;
}

}  // namespace ggs::omp
