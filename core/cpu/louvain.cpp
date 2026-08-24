// core/cpu/louvain.cpp — Tier T0 sequential Louvain.
#include "louvain.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

namespace ggs::seq {

namespace {

// A weighted undirected graph used across aggregation levels.
struct WGraph {
    std::size_t n = 0;
    std::vector<std::vector<std::pair<std::size_t, double>>> adj;  // neighbor, weight
    std::vector<double> selfloop;   // internal self-loop weight per node
    double total_weight = 0.0;      // m = sum of all edge weights (undirected)

    double weighted_degree(std::size_t v) const {
        double k = 2.0 * selfloop[v];
        for (auto& [u, w] : adj[v]) k += w;
        return k;
    }
};

// Build the level-0 weighted graph from the (symmetrized) CSR.
WGraph from_csr(const CsrGraph& g) {
    WGraph w;
    w.n = g.num_vertices;
    w.adj.resize(w.n);
    w.selfloop.assign(w.n, 0.0);
    for (VertexId u = 0; u < g.num_vertices; ++u)
        for (EdgeId e = g.row_offsets[u]; e < g.row_offsets[u + 1]; ++e) {
            VertexId v = g.col_indices[e];
            if (u == v) { w.selfloop[u] += 0.5; continue; }  // half, counted twice
            w.adj[u].emplace_back(v, 1.0);
        }
    // m = half the sum of weighted degrees.
    double deg_sum = 0.0;
    for (std::size_t v = 0; v < w.n; ++v) deg_sum += w.weighted_degree(v);
    w.total_weight = deg_sum / 2.0;
    return w;
}

double modularity(const WGraph& g, const std::vector<std::size_t>& comm) {
    const double m2 = 2.0 * g.total_weight;
    if (m2 == 0.0) return 0.0;
    std::unordered_map<std::size_t, double> in, tot;
    for (std::size_t v = 0; v < g.n; ++v) {
        tot[comm[v]] += g.weighted_degree(v);
        in[comm[v]]  += 2.0 * g.selfloop[v];
        for (auto& [u, wt] : g.adj[v])
            if (comm[u] == comm[v]) in[comm[v]] += wt;  // each internal edge counted twice over v,u
    }
    double q = 0.0;
    for (auto& [c, t] : tot) {
        double i = in.count(c) ? in[c] : 0.0;
        q += i / m2 - (t / m2) * (t / m2);
    }
    return q;
}

// One level: move nodes greedily to maximize modularity gain until stable.
// Returns the community label per node (not yet renumbered).
bool one_level(const WGraph& g, std::vector<std::size_t>& comm,
               int max_passes, double min_gain) {
    const double m2 = 2.0 * g.total_weight;
    std::vector<double> k(g.n), sigma_tot(g.n, 0.0);
    for (std::size_t v = 0; v < g.n; ++v) { k[v] = g.weighted_degree(v); comm[v] = v; }
    for (std::size_t v = 0; v < g.n; ++v) sigma_tot[comm[v]] += k[v];

    bool any_improvement = false;
    for (int pass = 0; pass < max_passes; ++pass) {
        bool moved = false;
        for (std::size_t v = 0; v < g.n; ++v) {
            // Weight from v to each neighboring community.
            std::unordered_map<std::size_t, double> k_in;
            for (auto& [u, wt] : g.adj[v])
                if (u != v) k_in[comm[u]] += wt;

            std::size_t old_c = comm[v];
            sigma_tot[old_c] -= k[v];
            double remove_cost = k_in.count(old_c) ? k_in[old_c] : 0.0;

            // Find the best community (default: stay in old).
            std::size_t best_c = old_c;
            double best_gain = 0.0;                      // gain relative to old
            double base = remove_cost - sigma_tot[old_c] * k[v] / m2;
            for (auto& [c, kin] : k_in) {
                double gain = kin - sigma_tot[c] * k[v] / m2 - base;
                if (gain > best_gain + min_gain) { best_gain = gain; best_c = c; }
            }
            sigma_tot[best_c] += k[v];
            comm[v] = best_c;
            if (best_c != old_c) { moved = true; any_improvement = true; }
        }
        if (!moved) break;
    }
    return any_improvement;
}

// Renumber arbitrary community ids to a dense 0..K-1 range.
std::size_t renumber(std::vector<std::size_t>& comm) {
    std::unordered_map<std::size_t, std::size_t> remap;
    for (auto& c : comm) {
        auto it = remap.find(c);
        if (it == remap.end()) { std::size_t id = remap.size(); remap[c] = id; c = id; }
        else c = it->second;
    }
    return remap.size();
}

// Aggregate: each community becomes a node in the next-level graph.
WGraph aggregate(const WGraph& g, const std::vector<std::size_t>& comm, std::size_t k) {
    WGraph ng;
    ng.n = k;
    ng.adj.resize(k);
    ng.selfloop.assign(k, 0.0);
    ng.total_weight = g.total_weight;   // modularity-preserving

    std::vector<std::unordered_map<std::size_t, double>> acc(k);
    for (std::size_t v = 0; v < g.n; ++v) {
        std::size_t cv = comm[v];
        ng.selfloop[cv] += g.selfloop[v];
        for (auto& [u, wt] : g.adj[v]) {
            std::size_t cu = comm[u];
            if (cu == cv) ng.selfloop[cv] += wt / 2.0;  // internal, halved (seen twice)
            else acc[cv][cu] += wt;
        }
    }
    for (std::size_t c = 0; c < k; ++c)
        for (auto& [d, wt] : acc[c]) ng.adj[c].emplace_back(d, wt);
    return ng;
}

}  // namespace

LouvainResult louvain(const CsrGraph& g, int max_passes, double min_gain) {
    LouvainResult r;
    if (g.num_vertices == 0) return r;

    WGraph cur = from_csr(g);
    // Track each original vertex's community as levels aggregate.
    std::vector<std::size_t> node_to_comm(g.num_vertices);
    std::iota(node_to_comm.begin(), node_to_comm.end(), 0);

    while (true) {
        std::vector<std::size_t> comm(cur.n);
        bool improved = one_level(cur, comm, max_passes, min_gain);
        std::size_t k = renumber(comm);

        // Fold this level's assignment into the original-vertex mapping.
        for (std::size_t v = 0; v < node_to_comm.size(); ++v)
            node_to_comm[v] = comm[node_to_comm[v]];

        r.levels++;
        if (!improved || k == cur.n) {          // converged
            cur = aggregate(cur, comm, k);
            break;
        }
        cur = aggregate(cur, comm, k);
    }

    r.num_communities = static_cast<VertexId>(cur.n);
    r.community.resize(g.num_vertices);
    for (VertexId v = 0; v < g.num_vertices; ++v)
        r.community[v] = static_cast<VertexId>(node_to_comm[v]);

    // Final modularity on the original graph under this labeling.
    WGraph base = from_csr(g);
    std::vector<std::size_t> c(g.num_vertices);
    for (VertexId v = 0; v < g.num_vertices; ++v) c[v] = node_to_comm[v];
    r.modularity = modularity(base, c);
    return r;
}

}  // namespace ggs::seq
