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

}  // namespace ggs::cuda
