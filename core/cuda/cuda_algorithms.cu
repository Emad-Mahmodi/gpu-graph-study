// core/cuda/cuda_algorithms.cu — Tier T2 naïve CUDA kernels.
//
// Design notes:
//  * CSR is uploaded once per call; results copied back to host std::vectors.
//  * One thread per vertex; global-memory access; no shared memory / warp
//    primitives (those are T3). This is the honest naïve baseline.
//  * Every algorithm mirrors the sequential oracle's semantics so results match
//    bit-for-bit (BFS, WCC) or within tol (PageRank).
#include "cuda_algorithms.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <string>

namespace ggs::cuda {

namespace {

constexpr int kBlock = 256;

// Degree thresholds for the T4 load-balancing tier, shared by PageRank, BFS and
// WCC so the three studies bucket identically:
//   deg <= kDegThresh              -> one thread per vertex
//   kDegThresh < deg <= kBlockThresh -> one warp  per vertex
//   deg >  kBlockThresh            -> one block per vertex
constexpr EdgeId kDegThresh   = 32;    // low/mid split — one warp's worth
constexpr EdgeId kBlockThresh = 1024;  // mid/high split — extreme hubs get a block

inline void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
        std::abort();
    }
}
#define CUDA_CHECK(x) check((x), #x)

inline int grid_for(std::uint32_t n) {
    return static_cast<int>((n + kBlock - 1) / kBlock);
}

// Small RAII helper for device buffers.
template <typename T>
struct DBuf {
    T* p = nullptr;
    explicit DBuf(std::size_t count) { CUDA_CHECK(cudaMalloc(&p, count * sizeof(T))); }
    ~DBuf() { if (p) cudaFree(p); }
    DBuf(const DBuf&) = delete;
    DBuf& operator=(const DBuf&) = delete;
};

// Upload the CSR topology (row offsets + column indices) to the device.
struct DeviceCsr {
    std::uint32_t n;
    std::uint64_t m;
    DBuf<EdgeId>   row;
    DBuf<VertexId> col;
    explicit DeviceCsr(const CsrGraph& g)
        : n(g.num_vertices), m(g.num_edges),
          row(g.num_vertices + 1), col(g.num_edges ? g.num_edges : 1) {
        CUDA_CHECK(cudaMemcpy(row.p, g.row_offsets.data(),
                              (n + 1) * sizeof(EdgeId), cudaMemcpyHostToDevice));
        if (g.num_edges)
            CUDA_CHECK(cudaMemcpy(col.p, g.col_indices.data(),
                                  m * sizeof(VertexId), cudaMemcpyHostToDevice));
    }
};

// ---------------- BFS ----------------------------------------------------
__global__ void bfs_step(const EdgeId* row, const VertexId* col,
                         std::int32_t* dist, int level,
                         std::uint32_t n, int* changed) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n || dist[v] != level) return;
    for (EdgeId e = row[v]; e < row[v + 1]; ++e) {
        VertexId u = col[e];
        if (dist[u] == -1) { dist[u] = level + 1; *changed = 1; }
    }
}

// ---------------- BFS T3 (frontier worklist) ----------------------------
// T2 launches one thread per vertex every level and most threads immediately
// return (dist[v] != level). T3 keeps an explicit frontier: only the vertices
// discovered last level are launched. A newly reached vertex is claimed exactly
// once with atomicCAS (-1 -> level+1) and appended to the next frontier via a
// single atomicAdd on the output counter. Distances are identical to T0.
__global__ void bfs_frontier_step(const EdgeId* row, const VertexId* col,
                                  std::int32_t* dist,
                                  const VertexId* fin, std::uint32_t in_count,
                                  VertexId* fout, int* out_count, int level) {
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= in_count) return;
    VertexId v = fin[i];
    for (EdgeId e = row[v]; e < row[v + 1]; ++e) {
        VertexId u = col[e];
        if (dist[u] != -1) continue;                       // fast path
        int old = atomicCAS(reinterpret_cast<int*>(&dist[u]), -1, level + 1);
        if (old == -1) {                                   // we claimed u
            int slot = atomicAdd(out_count, 1);
            fout[slot] = u;
        }
    }
}

// ---------------- BFS T4 (per-level degree-bucketed frontier) -----------
// T3 removed the wasted *launches* (only the frontier runs) but left the load
// imbalance inside a launch: one thread still walks a hub's whole neighbour
// list while its 31 warp-mates, holding degree-1 leaves, idle. T4 applies the
// same Gunrock/Merrill bucketing as PageRank — but PageRank can bucket once on
// the host because every vertex is active every iteration, whereas the BFS
// frontier changes shape at every level. So the split is redone on the device
// each level (`bfs_bucket_split`), which is the roadmap's "re-bucket between
// levels". The three expand kernels below share one claim helper, so the
// visited/append semantics are literally identical to T3 — only the mapping of
// threads to edges differs. Distances therefore match the T0 oracle exactly.

// Claim one neighbour: identical to the body of the T3 frontier kernel.
__device__ inline void bfs_claim(const VertexId* col, EdgeId e,
                                 std::int32_t* dist, int level,
                                 VertexId* fout, int* out_count) {
    VertexId u = col[e];
    if (dist[u] != -1) return;                          // fast path
    int old = atomicCAS(reinterpret_cast<int*>(&dist[u]), -1, level + 1);
    if (old == -1) fout[atomicAdd(out_count, 1)] = u;   // we claimed u
}

// Partition this level's frontier into three sub-frontiers by out-degree.
__global__ void bfs_bucket_split(const EdgeId* row, const VertexId* fin,
                                 std::uint32_t in_count,
                                 VertexId* b_low, VertexId* b_mid, VertexId* b_high,
                                 int* c_low, int* c_mid, int* c_high) {
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= in_count) return;
    VertexId v = fin[i];
    EdgeId deg = row[v + 1] - row[v];
    if (deg > kBlockThresh)    b_high[atomicAdd(c_high, 1)] = v;
    else if (deg > kDegThresh) b_mid[atomicAdd(c_mid, 1)]  = v;
    else                       b_low[atomicAdd(c_low, 1)]  = v;
}

// Low-degree sub-frontier: one thread per vertex (the T3 mapping).
__global__ void bfs_expand_thread(const EdgeId* row, const VertexId* col,
                                  std::int32_t* dist, const VertexId* ids,
                                  std::uint32_t count, VertexId* fout,
                                  int* out_count, int level) {
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    VertexId v = ids[i];
    for (EdgeId e = row[v]; e < row[v + 1]; ++e)
        bfs_claim(col, e, dist, level, fout, out_count);
}

// Mid-degree sub-frontier: one warp per vertex. The 32 lanes stride the
// neighbour list, so consecutive lanes read consecutive col[] entries —
// coalesced, unlike the one-thread-walks-the-list access of T3.
__global__ void bfs_expand_warp(const EdgeId* row, const VertexId* col,
                                std::int32_t* dist, const VertexId* ids,
                                std::uint32_t count, VertexId* fout,
                                int* out_count, int level) {
    std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    std::uint32_t w = tid >> 5;              // warp-uniform: one warp per vertex
    if (w >= count) return;
    VertexId v = ids[w];
    for (EdgeId e = row[v] + lane; e < row[v + 1]; e += 32)
        bfs_claim(col, e, dist, level, fout, out_count);
}

// High-degree sub-frontier: one whole block per extreme hub, launched with
// exactly `count` blocks. No reduction is needed here (unlike PageRank) — each
// thread claims its own neighbours independently.
__global__ void bfs_expand_block(const EdgeId* row, const VertexId* col,
                                 std::int32_t* dist, const VertexId* ids,
                                 std::uint32_t count, VertexId* fout,
                                 int* out_count, int level) {
    if (blockIdx.x >= count) return;
    VertexId v = ids[blockIdx.x];
    for (EdgeId e = row[v] + threadIdx.x; e < row[v + 1]; e += blockDim.x)
        bfs_claim(col, e, dist, level, fout, out_count);
}

// ---------------- PageRank ----------------------------------------------
__global__ void pr_dangling(const EdgeId* row, const double* rank,
                            std::uint32_t n, double* dangling) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    if (row[v + 1] - row[v] == 0) atomicAdd(dangling, rank[v]);
}

// Compute the per-iteration additive term on the device (1 thread), so the
// dangling mass never has to round-trip to the host.
__global__ void pr_makeadd(const double* dangling, double base, double damping,
                           std::uint32_t n, double* add) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *add = base + damping * (*dangling) / static_cast<double>(n);
}

__global__ void pr_update(const EdgeId* row, const VertexId* col,
                          const double* rank, double* next,
                          double damping, const double* add, std::uint32_t n,
                          double* diff) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    double sum = 0.0;
    for (EdgeId e = row[v]; e < row[v + 1]; ++e) {
        VertexId u = col[e];
        EdgeId du = row[u + 1] - row[u];
        if (du) sum += rank[u] / static_cast<double>(du);
    }
    double nv = *add + damping * sum;
    next[v] = nv;
    atomicAdd(diff, fabs(nv - rank[v]));
}

// ---------------- PageRank T3 (optimized) -------------------------------
// The naïve (T2) PageRank loses to the CPU not because of the math but because
// each iteration does two reductions — the dangling mass and the L1 change —
// as single-address atomicAdds hammered by all N threads. With N ~ 10^6 that
// serializes the whole GPU. T3 fixes exactly that:
//   (1) block-level reduction in shared memory → one atomicAdd per *block*
//       (256× fewer atomics), and
//   (2) the inverse out-degree is precomputed once, so the hot gather loop is a
//       fused-multiply-add (rank[u] * inv_deg[u]) instead of a global load +
//       double division every iteration.
// Nothing about the numerical result changes; it is diffed against T0.

// Block reduction of a per-thread value into *acc via one atomicAdd per block.
// blockDim.x must be a power of two ≤ 1024 (kBlock = 256 here).
__device__ inline void block_reduce_add(double val, double* acc) {
    __shared__ double s[kBlock];
    int t = threadIdx.x;
    s[t] = val;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    if (t == 0) atomicAdd(acc, s[0]);
}

// One-time: inv_deg[v] = 1/outdeg(v) (0 for dangling vertices).
__global__ void pr_precompute_inv(const EdgeId* row, std::uint32_t n,
                                  double* inv_deg) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    EdgeId d = row[v + 1] - row[v];
    inv_deg[v] = d ? 1.0 / static_cast<double>(d) : 0.0;
}

// Dangling mass via block reduction instead of N single-address atomics.
__global__ void pr_dangling_opt(const EdgeId* row, const double* rank,
                                std::uint32_t n, double* dangling) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    double local = 0.0;
    if (v < n && (row[v + 1] - row[v]) == 0) local = rank[v];
    block_reduce_add(local, dangling);
}

// Update + L1-change, with the change accumulated by block reduction and the
// gather using precomputed inverse degree (no per-iteration division).
__global__ void pr_update_opt(const EdgeId* row, const VertexId* col,
                              const double* rank, const double* inv_deg,
                              double* next, double damping, const double* add,
                              std::uint32_t n, double* diff) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    double local = 0.0;
    if (v < n) {
        double sum = 0.0;
        for (EdgeId e = row[v]; e < row[v + 1]; ++e) {
            VertexId u = col[e];
            sum += rank[u] * inv_deg[u];   // inv_deg[u] == 0 for dangling u
        }
        double nv = *add + damping * sum;
        next[v] = nv;
        local = fabs(nv - rank[v]);
    }
    block_reduce_add(local, diff);
}

// ---------------- PageRank T4 (degree-bucketed load balancing) ----------
// T3 fixed the reduction, but the *gather* is still one thread per vertex: on a
// power-law graph a single thread walks a 10^5-degree hub's neighbor list while
// its 31 warp-mates idle (load imbalance + no coalescing). T4 splits vertices by
// degree (computed once on the host) and processes each bucket with the right
// granularity — the established Gunrock/Merrill strategy:
//   * low-degree  (≤ kDegThresh): one thread per vertex (as T3).
//   * high-degree (>  kDegThresh): one warp per vertex — 32 lanes stride the
//     neighbor list (coalesced col[] reads) and combine via __shfl_down_sync.
// Both kernels write into next[] and accumulate the L1 change with the same
// block-level reduction as T3.
// Low-degree bucket: thread per vertex, indexed through the id list.
__global__ void pr_bucket_thread(const EdgeId* row, const VertexId* col,
                                 const double* rank, const double* inv_deg,
                                 double* next, double damping, const double* add,
                                 const VertexId* ids, std::uint32_t count,
                                 double* diff) {
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    double local = 0.0;
    if (i < count) {
        VertexId v = ids[i];
        double sum = 0.0;
        for (EdgeId e = row[v]; e < row[v + 1]; ++e)
            sum += rank[col[e]] * inv_deg[col[e]];
        double nv = *add + damping * sum;
        next[v] = nv;
        local = fabs(nv - rank[v]);
    }
    block_reduce_add(local, diff);
}

// High-degree bucket: warp per vertex. blockDim must be a multiple of 32.
__global__ void pr_bucket_warp(const EdgeId* row, const VertexId* col,
                               const double* rank, const double* inv_deg,
                               double* next, double damping, const double* add,
                               const VertexId* ids, std::uint32_t count,
                               double* diff) {
    std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    std::uint32_t w = tid >> 5;             // warp-uniform: one warp per vertex
    double local = 0.0;
    if (w < count) {
        VertexId v = ids[w];
        double sum = 0.0;
        for (EdgeId e = row[v] + lane; e < row[v + 1]; e += 32)
            sum += rank[col[e]] * inv_deg[col[e]];
        // Warp reduction of the 32 partial sums.
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, off);
        if (lane == 0) {
            double nv = *add + damping * sum;
            next[v] = nv;
            local = fabs(nv - rank[v]);
        }
    }
    block_reduce_add(local, diff);         // only lane 0 of each warp is nonzero
}

// High-degree bucket, level 3: one whole block per extreme hub. All blockDim
// threads stride the neighbor list; a shared-memory tree reduction combines the
// partial sums, then thread 0 writes next[v] and the L1 change. Launched with
// exactly `count` blocks, so blockIdx.x is always a valid hub index.
__global__ void pr_bucket_block(const EdgeId* row, const VertexId* col,
                                const double* rank, const double* inv_deg,
                                double* next, double damping, const double* add,
                                const VertexId* ids, std::uint32_t count,
                                double* diff) {
    __shared__ double s[kBlock];
    std::uint32_t bi = blockIdx.x;
    if (bi >= count) return;                    // safety; normally bi < count
    VertexId v = ids[bi];
    int t = threadIdx.x;
    double sum = 0.0;
    for (EdgeId e = row[v] + t; e < row[v + 1]; e += blockDim.x)
        sum += rank[col[e]] * inv_deg[col[e]];
    s[t] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    if (t == 0) {
        double nv = *add + damping * s[0];
        next[v] = nv;
        atomicAdd(diff, fabs(nv - rank[v]));    // one atomic per hub — negligible
    }
}

// ---------------- WCC (label propagation) -------------------------------
__global__ void wcc_init(VertexId* label, std::uint32_t n) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) label[v] = v;
}

__global__ void wcc_step(const EdgeId* row, const VertexId* col,
                         VertexId* label, std::uint32_t n, int* changed) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    VertexId best = label[v];
    for (EdgeId e = row[v]; e < row[v + 1]; ++e) {
        VertexId lu = label[col[e]];
        if (lu < best) best = lu;
    }
    if (best < label[v]) { label[v] = best; *changed = 1; }
}

// ---------------- WCC T3 (active-set label propagation) ------------------
// T2 sweeps all N vertices every round. T3 only reprocesses vertices that are
// "active" — those whose neighborhood changed last round. When a vertex lowers
// its label it marks its neighbors active for the next round (a benign race:
// every write stores the same value, 1). Labels are identical to T0.
__global__ void wcc_step_active(const EdgeId* row, const VertexId* col,
                                VertexId* label, const std::uint8_t* ain,
                                std::uint8_t* aout, std::uint32_t n,
                                int* changed) {
    std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n || !ain[v]) return;
    VertexId best = label[v];
    for (EdgeId e = row[v]; e < row[v + 1]; ++e) {
        VertexId lu = label[col[e]];
        if (lu < best) best = lu;
    }
    if (best < label[v]) {
        label[v] = best;
        *changed = 1;
        // Our lower label may let neighbors drop too — wake them next round.
        for (EdgeId e = row[v]; e < row[v + 1]; ++e) aout[col[e]] = 1;
    }
}

// ---------------- WCC T4 (degree-bucketed active-set) -------------------
// T3 fixed *which* vertices are rescanned (only the woken ones). It did not fix
// *how* a vertex is scanned: a thread owning a 10^5-degree hub still walks the
// whole list alone. T4 keeps the T3 active-set and adds the same three-level
// Gunrock bucketing used by PageRank. WCC's degrees are static, so the buckets
// are built once on the host (unlike BFS, whose frontier changes every level).
//
// Each bucket kernel does the same thing as `wcc_step_active`: take the minimum
// label over the neighbourhood, and if it lowers our own label, write it and
// wake the neighbours. Only the reduction differs — a warp shuffle for the mid
// bucket, a shared-memory tree for the high bucket. Label propagation converges
// to the same fixpoint (the minimum id per component) regardless of the order
// updates are applied in, so labels match T0 exactly.

// Low-degree bucket: one thread per vertex — the T3 mapping, over an id list.
__global__ void wcc_bucket_thread(const EdgeId* row, const VertexId* col,
                                  VertexId* label, const std::uint8_t* ain,
                                  std::uint8_t* aout, const VertexId* ids,
                                  std::uint32_t count, int* changed) {
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    VertexId v = ids[i];
    if (!ain[v]) return;
    VertexId my = label[v], best = my;
    for (EdgeId e = row[v]; e < row[v + 1]; ++e) {
        VertexId lu = label[col[e]];
        if (lu < best) best = lu;
    }
    if (best < my) {
        label[v] = best;
        *changed = 1;
        for (EdgeId e = row[v]; e < row[v + 1]; ++e) aout[col[e]] = 1;
    }
}

// Mid-degree bucket: one warp per vertex. Lanes stride the neighbour list
// (coalesced label[col[e]] gathers), then min-reduce via __shfl_down_sync and
// broadcast the winner back to every lane so the whole warp can do the
// neighbour wake-up pass, also coalesced.
__global__ void wcc_bucket_warp(const EdgeId* row, const VertexId* col,
                                VertexId* label, const std::uint8_t* ain,
                                std::uint8_t* aout, const VertexId* ids,
                                std::uint32_t count, int* changed) {
    std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    std::uint32_t w = tid >> 5;              // warp-uniform
    if (w >= count) return;
    VertexId v = ids[w];
    if (!ain[v]) return;                     // warp-uniform: v is warp-uniform
    // Read our own label *before* any lane writes it, so the comparison below
    // cannot race with lane 0's store.
    VertexId my = label[v], best = my;
    for (EdgeId e = row[v] + lane; e < row[v + 1]; e += 32) {
        VertexId lu = label[col[e]];
        if (lu < best) best = lu;
    }
    for (int off = 16; off > 0; off >>= 1) {
        VertexId o = __shfl_down_sync(0xffffffffu, best, off);
        if (o < best) best = o;
    }
    best = __shfl_sync(0xffffffffu, best, 0);   // broadcast the warp minimum
    if (best < my) {
        if (lane == 0) { label[v] = best; *changed = 1; }
        for (EdgeId e = row[v] + lane; e < row[v + 1]; e += 32) aout[col[e]] = 1;
    }
}

// High-degree bucket: one block per extreme hub, launched with exactly `count`
// blocks. Shared-memory tree min-reduction, then the whole block cooperates on
// the wake-up pass.
__global__ void wcc_bucket_block(const EdgeId* row, const VertexId* col,
                                 VertexId* label, const std::uint8_t* ain,
                                 std::uint8_t* aout, const VertexId* ids,
                                 std::uint32_t count, int* changed) {
    __shared__ VertexId s[kBlock];
    if (blockIdx.x >= count) return;
    VertexId v = ids[blockIdx.x];
    if (!ain[v]) return;                     // block-uniform
    int t = threadIdx.x;
    VertexId my = label[v], best = my;
    for (EdgeId e = row[v] + t; e < row[v + 1]; e += blockDim.x) {
        VertexId lu = label[col[e]];
        if (lu < best) best = lu;
    }
    s[t] = best;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (t < stride && s[t + stride] < s[t]) s[t] = s[t + stride];
        __syncthreads();
    }
    VertexId res = s[0];                     // valid in every thread after sync
    if (res < my) {
        if (t == 0) { label[v] = res; *changed = 1; }
        for (EdgeId e = row[v] + t; e < row[v + 1]; e += blockDim.x) aout[col[e]] = 1;
    }
}

}  // namespace

// ------------------------------------------------------------------------
bool available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return false;
    return count > 0;
}

std::string device_name() {
    if (!available()) return "";
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return "";
    return std::string(prop.name);
}

std::vector<std::int32_t> bfs(const CsrGraph& g, VertexId source) {
    const std::uint32_t n = g.num_vertices;
    std::vector<std::int32_t> host(n, -1);
    if (n == 0) return host;
    host[source] = 0;

    DeviceCsr d(g);
    DBuf<std::int32_t> dist(n);
    CUDA_CHECK(cudaMemcpy(dist.p, host.data(), n * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice));
    DBuf<int> changed(1);

    int level = 0, h_changed = 1;
    while (h_changed) {
        CUDA_CHECK(cudaMemset(changed.p, 0, sizeof(int)));
        bfs_step<<<grid_for(n), kBlock>>>(d.row.p, d.col.p, dist.p, level, n, changed.p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&h_changed, changed.p, sizeof(int), cudaMemcpyDeviceToHost));
        ++level;
    }
    CUDA_CHECK(cudaMemcpy(host.data(), dist.p, n * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    return host;
}

std::vector<std::int32_t> bfs_opt(const CsrGraph& g, VertexId source) {
    const std::uint32_t n = g.num_vertices;
    std::vector<std::int32_t> host(n, -1);
    if (n == 0) return host;
    host[source] = 0;

    DeviceCsr d(g);
    DBuf<std::int32_t> dist(n);
    CUDA_CHECK(cudaMemcpy(dist.p, host.data(), n * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice));

    DBuf<VertexId> fa(n), fb(n);
    DBuf<int> d_count(1);
    CUDA_CHECK(cudaMemcpy(fa.p, &source, sizeof(VertexId), cudaMemcpyHostToDevice));

    VertexId* fin = fa.p;
    VertexId* fout = fb.p;
    std::uint32_t in_count = 1;
    int level = 0;
    while (in_count > 0) {
        CUDA_CHECK(cudaMemset(d_count.p, 0, sizeof(int)));
        bfs_frontier_step<<<grid_for(in_count), kBlock>>>(
            d.row.p, d.col.p, dist.p, fin, in_count, fout, d_count.p, level);
        CUDA_CHECK(cudaGetLastError());
        int h_count = 0;
        CUDA_CHECK(cudaMemcpy(&h_count, d_count.p, sizeof(int), cudaMemcpyDeviceToHost));
        in_count = static_cast<std::uint32_t>(h_count);
        std::swap(fin, fout);
        ++level;
    }
    CUDA_CHECK(cudaMemcpy(host.data(), dist.p, n * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    return host;
}

std::vector<std::int32_t> bfs_lb(const CsrGraph& g, VertexId source) {
    const std::uint32_t n = g.num_vertices;
    std::vector<std::int32_t> host(n, -1);
    if (n == 0) return host;
    host[source] = 0;

    DeviceCsr d(g);
    DBuf<std::int32_t> dist(n);
    CUDA_CHECK(cudaMemcpy(dist.p, host.data(), n * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice));

    // Two ping-pong frontiers plus the three per-level sub-frontiers. A single
    // level can hold at most n vertices, so n is a safe bound for each.
    DBuf<VertexId> fa(n), fb(n), b_low(n), b_mid(n), b_high(n);
    // [0] = next-frontier size, [1..3] = low/mid/high bucket sizes.
    DBuf<int> d_counts(4);
    CUDA_CHECK(cudaMemcpy(fa.p, &source, sizeof(VertexId), cudaMemcpyHostToDevice));

    VertexId* fin  = fa.p;
    VertexId* fout = fb.p;
    std::uint32_t in_count = 1;
    int level = 0;
    const int warps_per_block = kBlock / 32;   // 8

    while (in_count > 0) {
        // --- 1. Re-bucket this level's frontier by degree, on the device. ---
        CUDA_CHECK(cudaMemset(d_counts.p, 0, 4 * sizeof(int)));
        bfs_bucket_split<<<grid_for(in_count), kBlock>>>(
            d.row.p, fin, in_count, b_low.p, b_mid.p, b_high.p,
            d_counts.p + 1, d_counts.p + 2, d_counts.p + 3);
        CUDA_CHECK(cudaGetLastError());

        // The bucket sizes decide the three grid dimensions, so they must come
        // back to the host before the expands launch. That is the price of
        // per-level bucketing: two device→host syncs per level instead of T3's
        // one — cheap on a shallow graph, visible on a high-diameter one.
        int h[4] = {0, 0, 0, 0};
        CUDA_CHECK(cudaMemcpy(h, d_counts.p, 4 * sizeof(int), cudaMemcpyDeviceToHost));
        const std::uint32_t low_n  = static_cast<std::uint32_t>(h[1]);
        const std::uint32_t mid_n  = static_cast<std::uint32_t>(h[2]);
        const std::uint32_t high_n = static_cast<std::uint32_t>(h[3]);

        // --- 2. Expand each bucket at its own granularity. ------------------
        if (low_n)
            bfs_expand_thread<<<grid_for(low_n), kBlock>>>(
                d.row.p, d.col.p, dist.p, b_low.p, low_n, fout, d_counts.p, level);
        if (mid_n)
            bfs_expand_warp<<<static_cast<int>((mid_n + warps_per_block - 1) / warps_per_block),
                              kBlock>>>(
                d.row.p, d.col.p, dist.p, b_mid.p, mid_n, fout, d_counts.p, level);
        if (high_n)
            bfs_expand_block<<<high_n, kBlock>>>(
                d.row.p, d.col.p, dist.p, b_high.p, high_n, fout, d_counts.p, level);
        CUDA_CHECK(cudaGetLastError());

        int h_out = 0;
        CUDA_CHECK(cudaMemcpy(&h_out, d_counts.p, sizeof(int), cudaMemcpyDeviceToHost));
        in_count = static_cast<std::uint32_t>(h_out);
        std::swap(fin, fout);
        ++level;
    }
    CUDA_CHECK(cudaMemcpy(host.data(), dist.p, n * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    return host;
}

std::vector<double> pagerank(const CsrGraph& g, double damping,
                             int max_iters, double tol) {
    const std::uint32_t n = g.num_vertices;
    std::vector<double> host(n, n ? 1.0 / n : 0.0);
    if (n == 0) return host;

    DeviceCsr d(g);
    DBuf<double> rank(n), next(n), d_dangling(1), d_add(1), d_diff(1);
    CUDA_CHECK(cudaMemcpy(rank.p, host.data(), n * sizeof(double),
                          cudaMemcpyHostToDevice));
    const double base = (1.0 - damping) / n;

    for (int it = 0; it < max_iters; ++it) {
        // Keep dangling mass + additive term entirely on the device — no host
        // round-trip. cudaMemset zeroes the accumulators (0x00 bytes == 0.0).
        CUDA_CHECK(cudaMemset(d_dangling.p, 0, sizeof(double)));
        pr_dangling<<<grid_for(n), kBlock>>>(d.row.p, rank.p, n, d_dangling.p);
        pr_makeadd<<<1, 1>>>(d_dangling.p, base, damping, n, d_add.p);

        CUDA_CHECK(cudaMemset(d_diff.p, 0, sizeof(double)));
        pr_update<<<grid_for(n), kBlock>>>(d.row.p, d.col.p, rank.p, next.p,
                                           damping, d_add.p, n, d_diff.p);
        CUDA_CHECK(cudaGetLastError());

        // The one unavoidable host sync per iteration: read the L1 change for
        // early stopping.
        double diff = 0.0;
        CUDA_CHECK(cudaMemcpy(&diff, d_diff.p, sizeof(double), cudaMemcpyDeviceToHost));
        std::swap(rank.p, next.p);
        if (diff < tol) break;
    }
    CUDA_CHECK(cudaMemcpy(host.data(), rank.p, n * sizeof(double), cudaMemcpyDeviceToHost));
    return host;
}

std::vector<double> pagerank_opt(const CsrGraph& g, double damping,
                                 int max_iters, double tol) {
    const std::uint32_t n = g.num_vertices;
    std::vector<double> host(n, n ? 1.0 / n : 0.0);
    if (n == 0) return host;

    DeviceCsr d(g);
    DBuf<double> rank(n), next(n), inv_deg(n), d_dangling(1), d_add(1), d_diff(1);
    CUDA_CHECK(cudaMemcpy(rank.p, host.data(), n * sizeof(double),
                          cudaMemcpyHostToDevice));
    const double base = (1.0 - damping) / n;

    // Precompute inverse out-degree once (T3 optimization #2).
    pr_precompute_inv<<<grid_for(n), kBlock>>>(d.row.p, n, inv_deg.p);
    CUDA_CHECK(cudaGetLastError());

    for (int it = 0; it < max_iters; ++it) {
        CUDA_CHECK(cudaMemset(d_dangling.p, 0, sizeof(double)));
        pr_dangling_opt<<<grid_for(n), kBlock>>>(d.row.p, rank.p, n, d_dangling.p);
        pr_makeadd<<<1, 1>>>(d_dangling.p, base, damping, n, d_add.p);

        CUDA_CHECK(cudaMemset(d_diff.p, 0, sizeof(double)));
        pr_update_opt<<<grid_for(n), kBlock>>>(d.row.p, d.col.p, rank.p,
                                               inv_deg.p, next.p, damping,
                                               d_add.p, n, d_diff.p);
        CUDA_CHECK(cudaGetLastError());

        double diff = 0.0;
        CUDA_CHECK(cudaMemcpy(&diff, d_diff.p, sizeof(double), cudaMemcpyDeviceToHost));
        std::swap(rank.p, next.p);
        if (diff < tol) break;
    }
    CUDA_CHECK(cudaMemcpy(host.data(), rank.p, n * sizeof(double), cudaMemcpyDeviceToHost));
    return host;
}

std::vector<double> pagerank_lb(const CsrGraph& g, double damping,
                                int max_iters, double tol) {
    const std::uint32_t n = g.num_vertices;
    std::vector<double> host(n, n ? 1.0 / n : 0.0);
    if (n == 0) return host;

    // Partition vertex ids into three degree buckets once (degrees are fixed
    // across iterations). This is the load-balancing decision, done on the host:
    //   low  (≤32)        -> thread per vertex
    //   mid  (33..1024)   -> warp per vertex
    //   high (>1024)      -> block per vertex
    std::vector<VertexId> low, mid, high;
    low.reserve(n);
    for (VertexId v = 0; v < n; ++v) {
        EdgeId deg = g.row_offsets[v + 1] - g.row_offsets[v];
        if (deg > kBlockThresh)     high.push_back(v);
        else if (deg > kDegThresh)  mid.push_back(v);
        else                        low.push_back(v);
    }
    const std::uint32_t low_n  = static_cast<std::uint32_t>(low.size());
    const std::uint32_t mid_n  = static_cast<std::uint32_t>(mid.size());
    const std::uint32_t high_n = static_cast<std::uint32_t>(high.size());

    DeviceCsr d(g);
    DBuf<double> rank(n), next(n), inv_deg(n), d_add(1), d_dangling(1), d_diff(1);
    CUDA_CHECK(cudaMemcpy(rank.p, host.data(), n * sizeof(double),
                          cudaMemcpyHostToDevice));
    DBuf<VertexId> d_low(low_n ? low_n : 1), d_mid(mid_n ? mid_n : 1),
                   d_high(high_n ? high_n : 1);
    if (low_n)  CUDA_CHECK(cudaMemcpy(d_low.p, low.data(),
                                      low_n * sizeof(VertexId), cudaMemcpyHostToDevice));
    if (mid_n)  CUDA_CHECK(cudaMemcpy(d_mid.p, mid.data(),
                                      mid_n * sizeof(VertexId), cudaMemcpyHostToDevice));
    if (high_n) CUDA_CHECK(cudaMemcpy(d_high.p, high.data(),
                                      high_n * sizeof(VertexId), cudaMemcpyHostToDevice));

    const double base = (1.0 - damping) / n;
    pr_precompute_inv<<<grid_for(n), kBlock>>>(d.row.p, n, inv_deg.p);
    CUDA_CHECK(cudaGetLastError());

    const int warps_per_block = kBlock / 32;   // 8
    const int low_grid = static_cast<int>((low_n + kBlock - 1) / kBlock);
    const int mid_grid = static_cast<int>((mid_n + warps_per_block - 1) / warps_per_block);

    for (int it = 0; it < max_iters; ++it) {
        CUDA_CHECK(cudaMemset(d_dangling.p, 0, sizeof(double)));
        pr_dangling_opt<<<grid_for(n), kBlock>>>(d.row.p, rank.p, n, d_dangling.p);
        pr_makeadd<<<1, 1>>>(d_dangling.p, base, damping, n, d_add.p);

        CUDA_CHECK(cudaMemset(d_diff.p, 0, sizeof(double)));
        if (low_n)
            pr_bucket_thread<<<low_grid, kBlock>>>(d.row.p, d.col.p, rank.p,
                inv_deg.p, next.p, damping, d_add.p, d_low.p, low_n, d_diff.p);
        if (mid_n)
            pr_bucket_warp<<<mid_grid, kBlock>>>(d.row.p, d.col.p, rank.p,
                inv_deg.p, next.p, damping, d_add.p, d_mid.p, mid_n, d_diff.p);
        if (high_n)
            pr_bucket_block<<<high_n, kBlock>>>(d.row.p, d.col.p, rank.p,
                inv_deg.p, next.p, damping, d_add.p, d_high.p, high_n, d_diff.p);
        CUDA_CHECK(cudaGetLastError());

        double diff = 0.0;
        CUDA_CHECK(cudaMemcpy(&diff, d_diff.p, sizeof(double), cudaMemcpyDeviceToHost));
        std::swap(rank.p, next.p);
        if (diff < tol) break;
    }
    CUDA_CHECK(cudaMemcpy(host.data(), rank.p, n * sizeof(double), cudaMemcpyDeviceToHost));
    return host;
}

std::vector<VertexId> wcc(const CsrGraph& g) {
    const std::uint32_t n = g.num_vertices;
    std::vector<VertexId> host(n);
    if (n == 0) return host;

    DeviceCsr d(g);
    DBuf<VertexId> label(n);
    DBuf<int> changed(1);
    wcc_init<<<grid_for(n), kBlock>>>(label.p, n);
    CUDA_CHECK(cudaGetLastError());

    int h_changed = 1;
    while (h_changed) {
        CUDA_CHECK(cudaMemset(changed.p, 0, sizeof(int)));
        wcc_step<<<grid_for(n), kBlock>>>(d.row.p, d.col.p, label.p, n, changed.p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&h_changed, changed.p, sizeof(int), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(host.data(), label.p, n * sizeof(VertexId), cudaMemcpyDeviceToHost));
    return host;
}

std::vector<VertexId> wcc_opt(const CsrGraph& g) {
    const std::uint32_t n = g.num_vertices;
    std::vector<VertexId> host(n);
    if (n == 0) return host;

    DeviceCsr d(g);
    DBuf<VertexId> label(n);
    DBuf<std::uint8_t> aa(n), ab(n);
    DBuf<int> changed(1);

    wcc_init<<<grid_for(n), kBlock>>>(label.p, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(aa.p, 1, n));   // round 1: everyone active

    std::uint8_t* ain = aa.p;
    std::uint8_t* aout = ab.p;
    int h_changed = 1;
    while (h_changed) {
        CUDA_CHECK(cudaMemset(aout, 0, n));
        CUDA_CHECK(cudaMemset(changed.p, 0, sizeof(int)));
        wcc_step_active<<<grid_for(n), kBlock>>>(d.row.p, d.col.p, label.p,
                                                 ain, aout, n, changed.p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&h_changed, changed.p, sizeof(int), cudaMemcpyDeviceToHost));
        std::swap(ain, aout);
    }
    CUDA_CHECK(cudaMemcpy(host.data(), label.p, n * sizeof(VertexId), cudaMemcpyDeviceToHost));
    return host;
}


std::vector<VertexId> wcc_lb(const CsrGraph& g) {
    const std::uint32_t n = g.num_vertices;
    std::vector<VertexId> host(n);
    if (n == 0) return host;

    // Degrees are static, so bucket once on the host (as in pagerank_lb).
    std::vector<VertexId> low, mid, high;
    low.reserve(n);
    for (VertexId v = 0; v < n; ++v) {
        EdgeId deg = g.row_offsets[v + 1] - g.row_offsets[v];
        if (deg > kBlockThresh)     high.push_back(v);
        else if (deg > kDegThresh)  mid.push_back(v);
        else                        low.push_back(v);
    }
    const std::uint32_t low_n  = static_cast<std::uint32_t>(low.size());
    const std::uint32_t mid_n  = static_cast<std::uint32_t>(mid.size());
    const std::uint32_t high_n = static_cast<std::uint32_t>(high.size());

    DeviceCsr d(g);
    DBuf<VertexId> label(n);
    DBuf<std::uint8_t> aa(n), ab(n);
    DBuf<int> changed(1);
    DBuf<VertexId> d_low(low_n ? low_n : 1), d_mid(mid_n ? mid_n : 1),
                   d_high(high_n ? high_n : 1);
    if (low_n)  CUDA_CHECK(cudaMemcpy(d_low.p, low.data(),
                                      low_n * sizeof(VertexId), cudaMemcpyHostToDevice));
    if (mid_n)  CUDA_CHECK(cudaMemcpy(d_mid.p, mid.data(),
                                      mid_n * sizeof(VertexId), cudaMemcpyHostToDevice));
    if (high_n) CUDA_CHECK(cudaMemcpy(d_high.p, high.data(),
                                      high_n * sizeof(VertexId), cudaMemcpyHostToDevice));

    wcc_init<<<grid_for(n), kBlock>>>(label.p, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(aa.p, 1, n));   // round 1: everyone active

    const int warps_per_block = kBlock / 32;   // 8
    const int low_grid = static_cast<int>((low_n + kBlock - 1) / kBlock);
    const int mid_grid = static_cast<int>((mid_n + warps_per_block - 1) / warps_per_block);

    std::uint8_t* ain  = aa.p;
    std::uint8_t* aout = ab.p;
    int h_changed = 1;
    while (h_changed) {
        CUDA_CHECK(cudaMemset(aout, 0, n));
        CUDA_CHECK(cudaMemset(changed.p, 0, sizeof(int)));
        if (low_n)
            wcc_bucket_thread<<<low_grid, kBlock>>>(d.row.p, d.col.p, label.p,
                ain, aout, d_low.p, low_n, changed.p);
        if (mid_n)
            wcc_bucket_warp<<<mid_grid, kBlock>>>(d.row.p, d.col.p, label.p,
                ain, aout, d_mid.p, mid_n, changed.p);
        if (high_n)
            wcc_bucket_block<<<high_n, kBlock>>>(d.row.p, d.col.p, label.p,
                ain, aout, d_high.p, high_n, changed.p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&h_changed, changed.p, sizeof(int), cudaMemcpyDeviceToHost));
        std::swap(ain, aout);
    }
    CUDA_CHECK(cudaMemcpy(host.data(), label.p, n * sizeof(VertexId), cudaMemcpyDeviceToHost));
    return host;
}

}  // namespace ggs::cuda
