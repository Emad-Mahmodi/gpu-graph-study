// core/cpu/louvain_omp.cpp â€” Tier T1 (OpenMP) Louvain.
//
// The T0 implementation stores each level as a vector<vector<pair>>; that is
// fine for a reference but is the wrong shape for a parallel loop (a pointer
// chase per vertex, and allocation churn from a per-vertex unordered_map).
// This tier keeps the same mathematics on a flat weighted-CSR level graph and
// replaces the per-vertex hash map with a dense per-thread scratch array plus a
// touched-list, so the inner loop is contiguous reads and no allocation.
//
// MSVC only implements OpenMP 2.0, so every parallel loop index is a signed
// long long and no OpenMP 3.1 constructs (atomic capture, collapse) are used.
#include "louvain_omp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ggs::omp {

namespace {

constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();

// A weighted undirected graph in flat CSR form, one per aggregation level.
// Self-loops are held separately (as in T0) because Louvain treats the internal
// weight of an aggregated community differently from its outgoing edges.
struct WCsr {
    std::size_t              n = 0;
    std::vector<std::size_t> off;        // n + 1
    std::vector<std::size_t> nbr;        // off[n]
    std::vector<double>      wgt;        // off[n]
    std::vector<double>      selfloop;   // n
    std::vector<double>      k;          // weighted degree (2*selfloop + row sum)
    double                   total_weight = 0.0;   // m
};

// Level 0: the symmetrized CSR with unit edge weights.
WCsr from_csr(const CsrGraph& g) {
    WCsr w;
    w.n = g.num_vertices;
    w.off.assign(w.n + 1, 0);
    w.selfloop.assign(w.n, 0.0);
    w.k.assign(w.n, 0.0);
    const long long n = static_cast<long long>(w.n);

    #pragma omp parallel for schedule(static)
    for (long long v = 0; v < n; ++v) {
        std::size_t cnt = 0;
        double sl = 0.0;
        for (EdgeId e = g.row_offsets[v]; e < g.row_offsets[v + 1]; ++e) {
            if (g.col_indices[e] == static_cast<VertexId>(v)) sl += 0.5;  // seen twice
            else ++cnt;
        }
        w.off[v + 1]  = cnt;
        w.selfloop[v] = sl;
    }
    for (std::size_t v = 0; v < w.n; ++v) w.off[v + 1] += w.off[v];   // prefix sum
    w.nbr.resize(w.off[w.n]);
    w.wgt.assign(w.off[w.n], 1.0);

    #pragma omp parallel for schedule(static)
    for (long long v = 0; v < n; ++v) {
        std::size_t p = w.off[v];
        for (EdgeId e = g.row_offsets[v]; e < g.row_offsets[v + 1]; ++e) {
            VertexId u = g.col_indices[e];
            if (u != static_cast<VertexId>(v)) w.nbr[p++] = u;
        }
    }

    double deg_sum = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:deg_sum)
    for (long long v = 0; v < n; ++v) {
        double kv = 2.0 * w.selfloop[v];
        for (std::size_t e = w.off[v]; e < w.off[v + 1]; ++e) kv += w.wgt[e];
        w.k[v] = kv;
        deg_sum += kv;
    }
    w.total_weight = deg_sum / 2.0;
    return w;
}

// Modularity of an assignment, same formula as T0. Accumulated with atomics on
// nc-sized arrays rather than per-thread copies, which would cost
// threads x nc doubles at level 0, where nc == n.
double modularity_of(const WCsr& g, const std::vector<std::size_t>& comm,
                     std::size_t nc) {
    const double m2 = 2.0 * g.total_weight;
    if (m2 == 0.0) return 0.0;
    std::vector<double> in(nc, 0.0), tot(nc, 0.0);
    const long long n = static_cast<long long>(g.n);

    #pragma omp parallel for schedule(dynamic, 256)
    for (long long v = 0; v < n; ++v) {
        const std::size_t cv = comm[v];
        double internal = 2.0 * g.selfloop[v];
        for (std::size_t e = g.off[v]; e < g.off[v + 1]; ++e)
            if (comm[g.nbr[e]] == cv) internal += g.wgt[e];   // counted twice over v,u
        #pragma omp atomic
        tot[cv] += g.k[v];
        #pragma omp atomic
        in[cv] += internal;
    }

    double q = 0.0;
    const long long knc = static_cast<long long>(nc);
    #pragma omp parallel for schedule(static) reduction(+:q)
    for (long long c = 0; c < knc; ++c)
        q += in[c] / m2 - (tot[c] / m2) * (tot[c] / m2);
    return q;
}

// How many chunks a pass is split into. This is the knob that trades quality
// for parallelism, and it is the single most consequential choice in this tier.
//
// With one chunk the pass is pure Jacobi: every vertex decides from the same
// all-singletons snapshot, so on a power-law graph they all pile onto the hubs
// at once and the level converges to a visibly worse partition. With n chunks
// it would be exactly Gauss-Seidel -- the oracle, with no parallelism left.
// Chunking sits between: inside a chunk the moves are computed from one
// snapshot and applied together, but the *next* chunk sees them, which restores
// most of the incremental visibility that makes sequential Louvain good.
//
// Measured on RMAT scale 17 (131k vertices, 4.2M edges), 8 threads, against the
// T0 oracle's 1840 ms / Q = 0.0792 -- `scripts/louvain_chunks.ps1`,
// `results/louvain_chunks.csv`:
//
//     chunks    time     Q        quality    speedup
//        1     240 ms   0.0575     72.6%      7.67x     pure Jacobi
//        4     507 ms   0.0741     93.6%      3.63x  <- chosen
//       16     900 ms   0.0745     94.1%      2.05x
//       64    1868 ms   0.0762     96.2%      0.99x     no advantage left
//      256    2860 ms   0.0765     96.6%      0.64x     slower than one core
//
// The knee is unambiguous: 1 -> 4 buys 21 points of quality for a 2.1x
// slowdown, while 4 -> 16 buys half a point for another 1.8x. By 64 the tier
// has no advantage left over the sequential oracle it is supposed to
// accelerate, and past that it is slower. So 4. (The Q column is reproducible
// exactly; only the timings move between runs.)
//
// kMinChunk keeps chunks wide enough that the threads still have real work on
// small graphs, where n / kChunks would otherwise be a few dozen vertices.
constexpr std::size_t kChunks   = 4;
constexpr std::size_t kMinChunk = 1024;

// One level of local moving. Returns whether modularity improved at all, and
// reports the level's final Q through q_out.
bool one_level(const WCsr& g, std::vector<std::size_t>& comm, int max_passes,
               double min_gain, double& q_out) {
    const std::size_t n = g.n;
    const double m2 = 2.0 * g.total_weight;
    comm.assign(n, 0);
    std::vector<double> sigma_tot(n, 0.0);
    const long long nn = static_cast<long long>(n);

    #pragma omp parallel for schedule(static)
    for (long long v = 0; v < nn; ++v) {
        comm[v]      = static_cast<std::size_t>(v);   // start as singletons
        sigma_tot[v] = g.k[v];
    }

    std::vector<std::size_t> target(n), csize(n), member(n), prev;
    double q_prev = modularity_of(g, comm, n);
    q_out = q_prev;
    if (m2 == 0.0) return false;
    bool improved = false;

    for (int pass = 0; pass < max_passes; ++pass) {
        // --- community sizes, and the sole member of each singleton ---------
        std::fill(csize.begin(), csize.end(), std::size_t{0});
        #pragma omp parallel for schedule(static)
        for (long long v = 0; v < nn; ++v) {
            const std::size_t c = comm[v];
            #pragma omp atomic
            csize[c]++;
            // Racy in general, but only read when csize[c] == 1, where the
            // writer is unique and the value is therefore correct.
            member[c] = static_cast<std::size_t>(v);
        }

        prev = comm;                                   // for the guard below
        long long moves = 0;

        // --- one chunk at a time: parallel inside, sequential between --------
        const std::size_t chunk = std::max(kMinChunk, (n + kChunks - 1) / kChunks);
        for (std::size_t lo = 0; lo < n; lo += chunk) {
            const long long a = static_cast<long long>(lo);
            const long long b = static_cast<long long>(std::min(lo + chunk, n));

            #pragma omp parallel
            {
                // Dense accumulator over communities plus the list of the ones
                // this vertex actually touched, so the reset is O(degree), not
                // O(n). Edge weights are strictly positive, so a zero entry
                // means "untouched" and no separate marker array is needed.
                std::vector<double> wacc(n, 0.0);
                std::vector<std::size_t> touched;

                #pragma omp for schedule(dynamic, 128)
                for (long long v = a; v < b; ++v) {
                    touched.clear();
                    for (std::size_t e = g.off[v]; e < g.off[v + 1]; ++e) {
                        const std::size_t c = comm[g.nbr[e]];
                        if (wacc[c] == 0.0) touched.push_back(c);
                        wacc[c] += g.wgt[e];
                    }
                    const std::size_t old_c = comm[v];
                    const double kv = g.k[v];
                    // Cost of leaving old_c, with this vertex removed from it.
                    const double base = wacc[old_c] - (sigma_tot[old_c] - kv) * kv / m2;

                    std::size_t best = old_c;
                    double best_gain = 0.0;
                    for (std::size_t i = 0; i < touched.size(); ++i) {
                        const std::size_t c = touched[i];
                        if (c == old_c) continue;
                        const double gain = wacc[c] - sigma_tot[c] * kv / m2 - base;
                        if (gain > best_gain + min_gain) { best_gain = gain; best = c; }
                    }
                    target[v] = best;
                    for (std::size_t i = 0; i < touched.size(); ++i) wacc[touched[i]] = 0.0;
                }
            }

            // --- break symmetric singleton swaps ----------------------------
            // Two isolated vertices that each want the other's community would
            // swap forever under simultaneous updates and never converge. The
            // standard fix (Lu et al. 2015): only the smaller id moves.
            #pragma omp parallel for schedule(static)
            for (long long v = a; v < b; ++v) {
                const std::size_t c = comm[v], t = target[v];
                if (t == c) continue;
                if (csize[c] == 1 && csize[t] == 1) {
                    const std::size_t u = member[t];
                    if (target[u] == c && static_cast<std::size_t>(v) > u) target[v] = c;
                }
            }

            // Apply this chunk's moves so the next chunk can see them, keeping
            // sigma_tot in step incrementally (a full recompute per chunk would
            // cost O(n) x kChunks per pass).
            #pragma omp parallel for schedule(static) reduction(+:moves)
            for (long long v = a; v < b; ++v) {
                const std::size_t old_c = comm[v], new_c = target[v];
                if (old_c == new_c) continue;
                const double kv = g.k[v];
                #pragma omp atomic
                sigma_tot[old_c] -= kv;
                #pragma omp atomic
                sigma_tot[new_c] += kv;
                comm[v] = new_c;
                ++moves;
            }
        }
        if (moves == 0) break;

        // --- modularity guard ------------------------------------------------
        // Gauss-Seidel Louvain improves Q by construction: each accepted move is
        // evaluated against the state it will actually land in. Jacobi has no
        // such guarantee -- a batch of individually-positive moves can be
        // jointly negative. So the pass is measured and rolled back if it did
        // not help, which makes this tier monotone in Q like the oracle. The
        // check costs one extra O(m) sweep per pass; that is the honest price of
        // the parallel formulation and it is included in every timing here.
        const double q = modularity_of(g, comm, n);
        if (q <= q_prev + min_gain) { comm.swap(prev); break; }
        q_prev   = q;
        improved = true;
    }

    q_out = q_prev;
    return improved;
}

// Dense-renumber community ids to 0..k-1. Serial: it is O(n) with no edge work,
// negligible next to a pass, and a parallel version would need a scan anyway.
std::size_t renumber(std::vector<std::size_t>& comm, std::size_t n) {
    std::vector<std::size_t> map(n, kNone);
    std::size_t k = 0;
    for (std::size_t v = 0; v < comm.size(); ++v) {
        const std::size_t c = comm[v];
        if (map[c] == kNone) map[c] = k++;
        comm[v] = map[c];
    }
    return k;
}

// Each community becomes one vertex of the next level. Vertices are grouped by
// community with a counting sort so every community's edge list can be built
// independently; the build is two passes (count, then fill) because the number
// of distinct neighbouring communities is not known in advance.
WCsr aggregate(const WCsr& g, const std::vector<std::size_t>& comm, std::size_t k) {
    std::vector<std::size_t> cstart(k + 1, 0), members(g.n);
    for (std::size_t v = 0; v < g.n; ++v) cstart[comm[v] + 1]++;
    for (std::size_t c = 0; c < k; ++c) cstart[c + 1] += cstart[c];
    {
        std::vector<std::size_t> cur(cstart.begin(), cstart.end());
        for (std::size_t v = 0; v < g.n; ++v) members[cur[comm[v]]++] = v;
    }

    WCsr ng;
    ng.n = k;
    ng.off.assign(k + 1, 0);
    ng.selfloop.assign(k, 0.0);
    ng.k.assign(k, 0.0);
    ng.total_weight = g.total_weight;         // aggregation preserves modularity
    const long long kk = static_cast<long long>(k);

    // Pass 1 -- how many distinct neighbouring communities does each one have,
    // and how much weight ends up inside it.
    #pragma omp parallel
    {
        std::vector<double> wacc(k, 0.0);
        std::vector<std::size_t> touched;
        #pragma omp for schedule(dynamic, 64)
        for (long long c = 0; c < kk; ++c) {
            touched.clear();
            double sl = 0.0;
            for (std::size_t i = cstart[c]; i < cstart[c + 1]; ++i) {
                const std::size_t v = members[i];
                sl += g.selfloop[v];
                for (std::size_t e = g.off[v]; e < g.off[v + 1]; ++e) {
                    const std::size_t cu = comm[g.nbr[e]];
                    if (cu == static_cast<std::size_t>(c)) { sl += g.wgt[e] / 2.0; continue; }
                    if (wacc[cu] == 0.0) touched.push_back(cu);
                    wacc[cu] += g.wgt[e];
                }
            }
            ng.selfloop[c] = sl;
            ng.off[c + 1]  = touched.size();
            for (std::size_t i = 0; i < touched.size(); ++i) wacc[touched[i]] = 0.0;
        }
    }
    for (std::size_t c = 0; c < k; ++c) ng.off[c + 1] += ng.off[c];
    ng.nbr.resize(ng.off[k]);
    ng.wgt.resize(ng.off[k]);

    // Pass 2 -- fill each community's row at its now-known offset.
    #pragma omp parallel
    {
        std::vector<double> wacc(k, 0.0);
        std::vector<std::size_t> touched;
        #pragma omp for schedule(dynamic, 64)
        for (long long c = 0; c < kk; ++c) {
            touched.clear();
            for (std::size_t i = cstart[c]; i < cstart[c + 1]; ++i) {
                const std::size_t v = members[i];
                for (std::size_t e = g.off[v]; e < g.off[v + 1]; ++e) {
                    const std::size_t cu = comm[g.nbr[e]];
                    if (cu == static_cast<std::size_t>(c)) continue;
                    if (wacc[cu] == 0.0) touched.push_back(cu);
                    wacc[cu] += g.wgt[e];
                }
            }
            std::size_t p = ng.off[c];
            double kc = 2.0 * ng.selfloop[c];
            for (std::size_t i = 0; i < touched.size(); ++i) {
                const std::size_t cu = touched[i];
                ng.nbr[p] = cu;
                ng.wgt[p] = wacc[cu];
                kc += wacc[cu];
                ++p;
                wacc[cu] = 0.0;
            }
            ng.k[c] = kc;
        }
    }
    return ng;
}

}  // namespace

seq::LouvainResult louvain(const CsrGraph& g, int max_passes, double min_gain) {
    seq::LouvainResult r;
    if (g.num_vertices == 0) return r;

    WCsr cur = from_csr(g);
    std::vector<std::size_t> node_to_comm(g.num_vertices);
    std::iota(node_to_comm.begin(), node_to_comm.end(), std::size_t{0});

    while (true) {
        std::vector<std::size_t> comm;
        double q_level = 0.0;
        const bool improved = one_level(cur, comm, max_passes, min_gain, q_level);
        const std::size_t k = renumber(comm, cur.n);

        const long long nn = static_cast<long long>(node_to_comm.size());
        #pragma omp parallel for schedule(static)
        for (long long v = 0; v < nn; ++v) node_to_comm[v] = comm[node_to_comm[v]];

        r.levels++;
        const bool done = (!improved || k == cur.n);
        cur = aggregate(cur, comm, k);
        if (done) break;
    }

    r.num_communities = static_cast<VertexId>(cur.n);
    r.community.resize(g.num_vertices);
    for (VertexId v = 0; v < g.num_vertices; ++v)
        r.community[v] = static_cast<VertexId>(node_to_comm[v]);

    // Final Q on the original graph, exactly as T0 reports it.
    WCsr base = from_csr(g);
    std::vector<std::size_t> c(g.num_vertices);
    for (VertexId v = 0; v < g.num_vertices; ++v) c[v] = node_to_comm[v];
    r.modularity = modularity_of(base, c, cur.n);
    return r;
}

}  // namespace ggs::omp
