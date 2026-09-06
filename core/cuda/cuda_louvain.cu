// core/cuda/cuda_louvain.cu - Tier T2, naive CUDA Louvain.
//
// One primitive carries the whole algorithm: label each arc with a 64-bit key,
// sort, then reduce equal keys. Both phases of Louvain reduce to it.
//
//   local moving   key = (owner vertex, community of the neighbour)
//                  -> weight from each vertex to each neighbouring community
//   aggregation    key = (community of u, community of v)
//                  -> the edge list of the next level's graph
//
// That is the naive choice on purpose. A per-warp hash table in shared memory
// is faster, but a hub with half a million neighbours overflows any shared
// table, and handling the overflow means building the degree-bucketed machinery
// that the T4 tier exists to isolate and measure. Sorting has no capacity limit
// and no special case, so this tier stays a clean baseline: obviously correct,
// predictably slow, and it makes the later speedup attributable to one named
// technique.
//
// Numerically this mirrors core/cpu/louvain.cpp exactly -- the same modularity
// gain, the same min-label rule against singleton swap oscillation, and the same
// guard that rolls back a pass which did not improve Q. It cannot reproduce the
// oracle's labels; sequential Louvain is order-dependent by construction.
#include "cuda_algorithms.hpp"

#include <cuda_runtime.h>

#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace ggs::cuda {

namespace {

constexpr int kBlk = 256;

inline void lcheck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
        std::abort();
    }
}
#define LCHECK(x) lcheck((x), #x)

// At least one block: aggregation legitimately produces a level with zero arcs
// (the tiny test graph reaches two disconnected communities), and a launch with
// a grid of 0 blocks is a hard "invalid configuration" error, not a no-op.
inline int lgrid(std::uint64_t n) {
    const std::uint64_t g = (n + kBlk - 1) / kBlk;
    return static_cast<int>(g ? g : 1);
}

// Owning device buffer. Deliberately separate from the DBuf in
// cuda_algorithms.cu: this file resizes buffers between levels, which that one
// never needs to do.
template <typename T>
struct Buf {
    T* p = nullptr;
    std::size_t cap = 0;
    Buf() = default;
    explicit Buf(std::size_t n) { resize(n); }
    ~Buf() { if (p) cudaFree(p); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    void resize(std::size_t n) {
        if (n <= cap && p) return;
        if (p) cudaFree(p);
        LCHECK(cudaMalloc(&p, (n ? n : 1) * sizeof(T)));
        cap = n ? n : 1;
    }
    void zero(std::size_t n) { LCHECK(cudaMemset(p, 0, n * sizeof(T))); }
};

// A weighted undirected graph on the device, one per aggregation level.
struct Level {
    std::uint32_t n = 0;
    std::uint64_t m = 0;          // directed arcs
    Buf<std::uint64_t> off;       // n + 1
    Buf<std::uint32_t> nbr;       // m
    Buf<double>        wgt;       // m
    Buf<double>        selfloop;  // n
    Buf<double>        k;         // n, weighted degree = 2*selfloop + row sum
    double total_weight = 0.0;    // m in the modularity formula
};

// ---------------------------------------------------------------- helpers --

// arc -> its owning vertex, so an arc can carry its own segment id into a sort.
__global__ void k_arc_owner(const std::uint64_t* off, std::uint32_t n,
                            std::uint32_t* owner) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    for (std::uint64_t e = off[v]; e < off[v + 1]; ++e) owner[e] = v;
}

// key = (owner << 32) | community(neighbour). Sorting this groups every arc by
// vertex first and by neighbouring community second, which is exactly the
// grouping the modularity gain needs.
__global__ void k_pair_keys(const std::uint32_t* owner, const std::uint32_t* nbr,
                            const std::uint32_t* comm, std::uint64_t m,
                            std::uint64_t* keys) {
    std::uint64_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= m) return;
    keys[e] = (static_cast<std::uint64_t>(owner[e]) << 32) | comm[nbr[e]];
}

__global__ void k_init_comm(std::uint32_t* comm, double* sigma_tot,
                            const double* k, std::uint32_t n) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    comm[v] = v;                 // every vertex starts as its own community
    sigma_tot[v] = k[v];
}

// Community sizes, plus the sole member of each singleton. `member` is racy in
// general but only read where csize == 1, and there the writer is unique.
__global__ void k_csize(const std::uint32_t* comm, std::uint32_t n,
                        std::uint32_t* csize, std::uint32_t* member) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    atomicAdd(&csize[comm[v]], 1u);
    member[comm[v]] = v;
}

// Walk the reduced (vertex, community) -> weight runs and pick each vertex's
// best move. `pk`/`pw` are sorted by key, so every entry for one vertex is
// contiguous; `pstart[v]` is where that vertex's block begins.
__global__ void k_choose(const std::uint64_t* pk, const double* pw,
                         const std::uint64_t* pstart, std::uint32_t n,
                         const std::uint32_t* comm, const double* sigma_tot,
                         const double* k, double m2, double min_gain,
                         std::uint32_t* target) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    const std::uint64_t lo = pstart[v], hi = pstart[v + 1];
    const std::uint32_t old_c = comm[v];
    const double kv = k[v];

    // Weight from v into its own community, if that community appears at all.
    double k_in_old = 0.0;
    for (std::uint64_t i = lo; i < hi; ++i)
        if (static_cast<std::uint32_t>(pk[i] & 0xFFFFFFFFull) == old_c) {
            k_in_old = pw[i];
            break;
        }
    // Cost of leaving old_c, with this vertex's own degree taken back out.
    const double base = k_in_old - (sigma_tot[old_c] - kv) * kv / m2;

    std::uint32_t best = old_c;
    double best_gain = 0.0;
    for (std::uint64_t i = lo; i < hi; ++i) {
        const std::uint32_t c = static_cast<std::uint32_t>(pk[i] & 0xFFFFFFFFull);
        if (c == old_c) continue;
        const double gain = pw[i] - sigma_tot[c] * kv / m2 - base;
        if (gain > best_gain + min_gain) { best_gain = gain; best = c; }
    }
    target[v] = best;
}

// Two singleton vertices that each want the other's community would swap
// forever under simultaneous updates. Standard fix: only the smaller id moves.
__global__ void k_break_swaps(const std::uint32_t* comm, std::uint32_t* target,
                              const std::uint32_t* csize,
                              const std::uint32_t* member, std::uint32_t n) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    const std::uint32_t c = comm[v], t = target[v];
    if (t == c) return;
    if (csize[c] == 1 && csize[t] == 1) {
        const std::uint32_t u = member[t];
        if (target[u] == c && v > u) target[v] = c;
    }
}

__global__ void k_apply(std::uint32_t* comm, const std::uint32_t* target,
                        const double* k, double* sigma_tot, std::uint32_t n,
                        unsigned long long* moves) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    const std::uint32_t old_c = comm[v], new_c = target[v];
    if (old_c == new_c) return;
    atomicAdd(&sigma_tot[old_c], -k[v]);
    atomicAdd(&sigma_tot[new_c],  k[v]);
    comm[v] = new_c;
    atomicAdd(moves, 1ull);
}

// in_c (edges inside a community, counted twice) and tot_c (sum of degrees).
__global__ void k_modularity_acc(const std::uint64_t* off, const std::uint32_t* nbr,
                                 const double* wgt, const double* selfloop,
                                 const double* k, const std::uint32_t* comm,
                                 std::uint32_t n, double* in, double* tot) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    const std::uint32_t cv = comm[v];
    double internal = 2.0 * selfloop[v];
    for (std::uint64_t e = off[v]; e < off[v + 1]; ++e)
        if (comm[nbr[e]] == cv) internal += wgt[e];
    atomicAdd(&tot[cv], k[v]);
    atomicAdd(&in[cv], internal);
}

__global__ void k_modularity_sum(const double* in, const double* tot,
                                 std::uint32_t nc, double m2, double* q) {
    __shared__ double s[kBlk];
    std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = threadIdx.x;
    double local = 0.0;
    if (c < nc) {
        const double tc = tot[c] / m2;
        local = in[c] / m2 - tc * tc;
    }
    s[t] = local;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    if (t == 0) atomicAdd(q, s[0]);
}

// Mark the first occurrence of each community id, so a prefix sum over the
// flags renumbers them densely.
__global__ void k_mark_used(const std::uint32_t* comm, std::uint32_t n,
                            std::uint32_t* used) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) used[comm[v]] = 1;
}
__global__ void k_apply_renumber(std::uint32_t* comm, const std::uint32_t* dense,
                                 std::uint32_t n) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) comm[v] = dense[comm[v]];
}
// Fold this level's assignment into the original-vertex mapping.
__global__ void k_fold(std::uint32_t* node_to_comm, const std::uint32_t* comm,
                       std::uint32_t n) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) node_to_comm[v] = comm[node_to_comm[v]];
}

// Aggregation: key = (community of u, community of v).
__global__ void k_agg_keys(const std::uint32_t* owner, const std::uint32_t* nbr,
                           const std::uint32_t* comm, std::uint64_t m,
                           std::uint64_t* keys) {
    std::uint64_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= m) return;
    keys[e] = (static_cast<std::uint64_t>(comm[owner[e]]) << 32) | comm[nbr[e]];
}

// Each community inherits the self-loops of its members.
__global__ void k_agg_selfloop_from_members(const double* selfloop,
                                            const std::uint32_t* comm,
                                            std::uint32_t n, double* out) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) atomicAdd(&out[comm[v]], selfloop[v]);
}

// Split the reduced (cu, cv) pairs: cu == cv becomes internal weight (halved,
// since each internal edge is seen from both ends), the rest become arcs.
__global__ void k_agg_split(const std::uint64_t* pk, const double* pw,
                            std::uint64_t np, double* selfloop,
                            std::uint32_t* degree) {
    std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np) return;
    const std::uint32_t cu = static_cast<std::uint32_t>(pk[i] >> 32);
    const std::uint32_t cv = static_cast<std::uint32_t>(pk[i] & 0xFFFFFFFFull);
    if (cu == cv) atomicAdd(&selfloop[cu], pw[i] * 0.5);
    else          atomicAdd(&degree[cu], 1u);
}

// Scatter the surviving pairs into the new CSR, using a moving cursor per row.
__global__ void k_agg_fill(const std::uint64_t* pk, const double* pw,
                           std::uint64_t np, const std::uint64_t* off,
                           std::uint32_t* cursor, std::uint32_t* nbr,
                           double* wgt) {
    std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np) return;
    const std::uint32_t cu = static_cast<std::uint32_t>(pk[i] >> 32);
    const std::uint32_t cv = static_cast<std::uint32_t>(pk[i] & 0xFFFFFFFFull);
    if (cu == cv) return;
    const std::uint32_t slot = atomicAdd(&cursor[cu], 1u);
    const std::uint64_t pos = off[cu] + slot;
    nbr[pos] = cv;
    wgt[pos] = pw[i];
}

__global__ void k_degrees(const std::uint64_t* off, const double* wgt,
                          const double* selfloop, std::uint32_t n, double* k) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    double kv = 2.0 * selfloop[v];
    for (std::uint64_t e = off[v]; e < off[v + 1]; ++e) kv += wgt[e];
    k[v] = kv;
}

// ------------------------------------------------------------- primitives --

// Sort the arc keys and sum equal ones. Returns the number of distinct pairs.
// out_keys / out_vals must hold at least m entries.
std::uint64_t sort_and_reduce(std::uint64_t* keys, double* vals, std::uint64_t m,
                              std::uint64_t* out_keys, double* out_vals) {
    if (m == 0) return 0;
    thrust::device_ptr<std::uint64_t> k(keys), ok(out_keys);
    thrust::device_ptr<double> v(vals), ov(out_vals);
    thrust::sort_by_key(k, k + m, v);
    auto end = thrust::reduce_by_key(k, k + m, v, ok, ov);
    return static_cast<std::uint64_t>(end.first - ok);
}

// Exclusive prefix sum in place over n+1 entries (offsets from counts).
void exclusive_scan_u64(std::uint64_t* d, std::uint32_t n) {
    thrust::device_ptr<std::uint64_t> p(d);
    thrust::exclusive_scan(p, p + n + 1, p);
}

double modularity_of(const Level& lv, const std::uint32_t* comm, std::uint32_t nc) {
    const double m2 = 2.0 * lv.total_weight;
    if (m2 == 0.0) return 0.0;
    Buf<double> in(nc), tot(nc), q(1);
    in.zero(nc); tot.zero(nc); q.zero(1);
    k_modularity_acc<<<lgrid(lv.n), kBlk>>>(lv.off.p, lv.nbr.p, lv.wgt.p,
                                            lv.selfloop.p, lv.k.p, comm, lv.n,
                                            in.p, tot.p);
    k_modularity_sum<<<lgrid(nc), kBlk>>>(in.p, tot.p, nc, m2, q.p);
    LCHECK(cudaGetLastError());
    double h = 0.0;
    LCHECK(cudaMemcpy(&h, q.p, sizeof(double), cudaMemcpyDeviceToHost));
    return h;
}

// ---------------------------------------------------------- level pipeline --

// Local moving on one level. Fills `comm`, returns whether Q improved.
bool one_level(const Level& lv, std::uint32_t* comm, int max_passes,
               double min_gain) {
    const std::uint32_t n = lv.n;
    const std::uint64_t m = lv.m;
    const double m2 = 2.0 * lv.total_weight;
    if (n == 0 || m2 == 0.0) return false;

    Buf<double> sigma_tot(n);
    Buf<std::uint32_t> target(n), csize(n), member(n), prev(n), owner(m);
    Buf<std::uint64_t> keys(m), pk(m), pstart(static_cast<std::size_t>(n) + 1);
    Buf<double> vals(m), pw(m);
    Buf<unsigned long long> moves(1);

    k_init_comm<<<lgrid(n), kBlk>>>(comm, sigma_tot.p, lv.k.p, n);
    k_arc_owner<<<lgrid(n), kBlk>>>(lv.off.p, n, owner.p);
    LCHECK(cudaGetLastError());

    double q_prev = modularity_of(lv, comm, n);
    bool improved = false;

    for (int pass = 0; pass < max_passes; ++pass) {
        csize.zero(n);
        k_csize<<<lgrid(n), kBlk>>>(comm, n, csize.p, member.p);

        // Group every arc by (vertex, neighbour community) and sum the runs.
        k_pair_keys<<<lgrid(m), kBlk>>>(owner.p, lv.nbr.p, comm, m, keys.p);
        LCHECK(cudaMemcpy(vals.p, lv.wgt.p, m * sizeof(double),
                          cudaMemcpyDeviceToDevice));
        LCHECK(cudaGetLastError());
        const std::uint64_t np = sort_and_reduce(keys.p, vals.p, m, pk.p, pw.p);

        // Where each vertex's run of pairs begins. The pairs are sorted by
        // (vertex, community), so a binary search on the vertex half of the key
        // gives the segment bounds.
        {
            thrust::device_ptr<std::uint64_t> pkp(pk.p), sp(pstart.p);
            std::vector<std::uint64_t> bounds(static_cast<std::size_t>(n) + 1);
            for (std::uint32_t v = 0; v <= n; ++v)
                bounds[v] = static_cast<std::uint64_t>(v) << 32;
            Buf<std::uint64_t> dbounds(static_cast<std::size_t>(n) + 1);
            LCHECK(cudaMemcpy(dbounds.p, bounds.data(),
                              (static_cast<std::size_t>(n) + 1) * sizeof(std::uint64_t),
                              cudaMemcpyHostToDevice));
            thrust::device_ptr<std::uint64_t> bp(dbounds.p);
            thrust::lower_bound(pkp, pkp + np, bp, bp + n + 1, sp);
        }

        k_choose<<<lgrid(n), kBlk>>>(pk.p, pw.p, pstart.p, n, comm, sigma_tot.p,
                                     lv.k.p, m2, min_gain, target.p);
        k_break_swaps<<<lgrid(n), kBlk>>>(comm, target.p, csize.p, member.p, n);
        LCHECK(cudaGetLastError());

        LCHECK(cudaMemcpy(prev.p, comm, n * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToDevice));
        moves.zero(1);
        k_apply<<<lgrid(n), kBlk>>>(comm, target.p, lv.k.p, sigma_tot.p, n, moves.p);
        LCHECK(cudaGetLastError());

        unsigned long long h_moves = 0;
        LCHECK(cudaMemcpy(&h_moves, moves.p, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
        if (h_moves == 0) break;

        // Simultaneous moves can be jointly negative even when each is
        // individually positive, so measure and roll back rather than assume.
        const double q = modularity_of(lv, comm, n);
        if (q <= q_prev + min_gain) {
            LCHECK(cudaMemcpy(comm, prev.p, n * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToDevice));
            break;
        }
        q_prev = q;
        improved = true;
    }
    return improved;
}

// Dense-renumber community ids to 0..k-1 and report k.
std::uint32_t renumber(std::uint32_t* comm, std::uint32_t n) {
    Buf<std::uint32_t> used(n), dense(n);
    used.zero(n);
    k_mark_used<<<lgrid(n), kBlk>>>(comm, n, used.p);
    LCHECK(cudaGetLastError());
    thrust::device_ptr<std::uint32_t> u(used.p), d(dense.p);
    thrust::exclusive_scan(u, u + n, d);
    k_apply_renumber<<<lgrid(n), kBlk>>>(comm, dense.p, n);
    LCHECK(cudaGetLastError());
    std::uint32_t last_flag = 0, last_pos = 0;
    LCHECK(cudaMemcpy(&last_flag, used.p + (n - 1), sizeof(std::uint32_t),
                      cudaMemcpyDeviceToHost));
    LCHECK(cudaMemcpy(&last_pos, dense.p + (n - 1), sizeof(std::uint32_t),
                      cudaMemcpyDeviceToHost));
    return last_pos + last_flag;
}

// Contract each community into one vertex of the next level.
void aggregate(const Level& in, const std::uint32_t* comm, std::uint32_t k,
               Level& out) {
    out.n = k;
    out.total_weight = in.total_weight;      // aggregation preserves modularity
    out.selfloop.resize(k);
    out.k.resize(k);
    out.off.resize(static_cast<std::size_t>(k) + 1);
    out.selfloop.zero(k);

    Buf<std::uint32_t> owner(in.m ? in.m : 1), degree(k);
    Buf<std::uint64_t> keys(in.m ? in.m : 1), pk(in.m ? in.m : 1);
    Buf<double> vals(in.m ? in.m : 1), pw(in.m ? in.m : 1);
    degree.zero(k);

    k_agg_selfloop_from_members<<<lgrid(in.n), kBlk>>>(in.selfloop.p, comm,
                                                       in.n, out.selfloop.p);
    k_arc_owner<<<lgrid(in.n), kBlk>>>(in.off.p, in.n, owner.p);
    k_agg_keys<<<lgrid(in.m), kBlk>>>(owner.p, in.nbr.p, comm, in.m, keys.p);
    LCHECK(cudaMemcpy(vals.p, in.wgt.p, in.m * sizeof(double),
                      cudaMemcpyDeviceToDevice));
    LCHECK(cudaGetLastError());

    const std::uint64_t np = sort_and_reduce(keys.p, vals.p, in.m, pk.p, pw.p);
    k_agg_split<<<lgrid(np), kBlk>>>(pk.p, pw.p, np, out.selfloop.p, degree.p);
    LCHECK(cudaGetLastError());

    // degree -> offsets. The counts live in [1..k] so the scan yields offsets.
    {
        std::vector<std::uint32_t> hdeg(k);
        LCHECK(cudaMemcpy(hdeg.data(), degree.p, k * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost));
        std::vector<std::uint64_t> hoff(static_cast<std::size_t>(k) + 1, 0);
        for (std::uint32_t c = 0; c < k; ++c) hoff[c + 1] = hoff[c] + hdeg[c];
        out.m = hoff[k];
        LCHECK(cudaMemcpy(out.off.p, hoff.data(),
                          (static_cast<std::size_t>(k) + 1) * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice));
    }
    out.nbr.resize(out.m ? out.m : 1);
    out.wgt.resize(out.m ? out.m : 1);

    Buf<std::uint32_t> cursor(k);
    cursor.zero(k);
    if (np) k_agg_fill<<<lgrid(np), kBlk>>>(pk.p, pw.p, np, out.off.p, cursor.p,
                                            out.nbr.p, out.wgt.p);
    k_degrees<<<lgrid(k), kBlk>>>(out.off.p, out.wgt.p, out.selfloop.p, k, out.k.p);
    LCHECK(cudaGetLastError());
}

// Upload the symmetrized CSR as level 0, with unit edge weights.
void upload_level0(const CsrGraph& g, Level& lv) {
    const std::uint32_t n = g.num_vertices;
    // T0 counts each directed self-loop arc as 0.5 of internal weight; mirror it.
    std::vector<std::uint64_t> off(static_cast<std::size_t>(n) + 1, 0);
    std::vector<std::uint32_t> nbr;
    std::vector<double> selfloop(n, 0.0);
    nbr.reserve(g.num_edges);
    for (VertexId v = 0; v < n; ++v) {
        for (EdgeId e = g.row_offsets[v]; e < g.row_offsets[v + 1]; ++e) {
            VertexId u = g.col_indices[e];
            if (u == v) selfloop[v] += 0.5;
            else        nbr.push_back(u);
        }
        off[v + 1] = nbr.size();
    }
    lv.n = n;
    lv.m = nbr.size();
    lv.off.resize(static_cast<std::size_t>(n) + 1);
    lv.nbr.resize(lv.m ? lv.m : 1);
    lv.wgt.resize(lv.m ? lv.m : 1);
    lv.selfloop.resize(n);
    lv.k.resize(n);

    LCHECK(cudaMemcpy(lv.off.p, off.data(),
                      (static_cast<std::size_t>(n) + 1) * sizeof(std::uint64_t),
                      cudaMemcpyHostToDevice));
    if (lv.m)
        LCHECK(cudaMemcpy(lv.nbr.p, nbr.data(), lv.m * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    LCHECK(cudaMemcpy(lv.selfloop.p, selfloop.data(), n * sizeof(double),
                      cudaMemcpyHostToDevice));
    {
        std::vector<double> ones(lv.m ? lv.m : 1, 1.0);
        LCHECK(cudaMemcpy(lv.wgt.p, ones.data(),
                          (lv.m ? lv.m : 1) * sizeof(double),
                          cudaMemcpyHostToDevice));
    }
    k_degrees<<<lgrid(n), kBlk>>>(lv.off.p, lv.wgt.p, lv.selfloop.p, n, lv.k.p);
    LCHECK(cudaGetLastError());

    thrust::device_ptr<double> kp(lv.k.p);
    lv.total_weight = thrust::reduce(kp, kp + n, 0.0) / 2.0;
}

}  // namespace

// ---------------------------------------------------------------- driver ---

seq::LouvainResult louvain(const CsrGraph& g, int max_passes, double min_gain) {
    seq::LouvainResult r;
    const std::uint32_t n0 = g.num_vertices;
    if (n0 == 0) return r;

    Level base, cur, next;
    upload_level0(g, base);
    upload_level0(g, cur);

    Buf<std::uint32_t> node_to_comm(n0);
    {
        std::vector<std::uint32_t> ident(n0);
        for (std::uint32_t v = 0; v < n0; ++v) ident[v] = v;
        LCHECK(cudaMemcpy(node_to_comm.p, ident.data(),
                          n0 * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    }

    while (true) {
        Buf<std::uint32_t> comm(cur.n);
        const bool improved = one_level(cur, comm.p, max_passes, min_gain);
        const std::uint32_t k = renumber(comm.p, cur.n);

        k_fold<<<lgrid(n0), kBlk>>>(node_to_comm.p, comm.p, n0);
        LCHECK(cudaGetLastError());
        r.levels++;

        const bool done = (!improved || k == cur.n);
        aggregate(cur, comm.p, k, next);
        // Move next -> cur. Buf is non-copyable, so swap the raw handles.
        std::swap(cur.n, next.n);           std::swap(cur.m, next.m);
        std::swap(cur.off.p, next.off.p);   std::swap(cur.off.cap, next.off.cap);
        std::swap(cur.nbr.p, next.nbr.p);   std::swap(cur.nbr.cap, next.nbr.cap);
        std::swap(cur.wgt.p, next.wgt.p);   std::swap(cur.wgt.cap, next.wgt.cap);
        std::swap(cur.selfloop.p, next.selfloop.p);
        std::swap(cur.selfloop.cap, next.selfloop.cap);
        std::swap(cur.k.p, next.k.p);       std::swap(cur.k.cap, next.k.cap);
        cur.total_weight = next.total_weight;
        if (done) break;
    }

    r.num_communities = cur.n;
    r.community.resize(n0);
    LCHECK(cudaMemcpy(r.community.data(), node_to_comm.p,
                      n0 * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    // Final Q on the original graph under this labelling, exactly as T0 reports.
    r.modularity = modularity_of(base, node_to_comm.p, cur.n);
    return r;
}

}  // namespace ggs::cuda
