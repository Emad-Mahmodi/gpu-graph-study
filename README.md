# A Reproducible Empirical Study of GPU Graph-Analytics Strategies on Irregular Security Graphs

**One graph-analytics engine, taken from a sequential CPU baseline through OpenMP and CUDA to Tensor-Core-era optimizations, benchmarked with a single honest methodology on highly skewed, power-law graphs — the kind produced by binary call-graphs / CFGs and by network-flow intrusion-detection data.**

This is an **empirical study**, not a claim of a new algorithm. Its value — like the survey it is modeled on [4] — is a clean taxonomy of optimization tiers, correct implementations of each, and a reproducible benchmark that attributes every speedup to a *specific* technique rather than a rewrite.

<p align="center">
  <a href="#results">Results</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#design">Design</a> ·
  <a href="#reproducibility">Reproducibility</a> ·
  <a href="#roadmap">Roadmap</a> ·
  <a href="#references">References</a>
</p>

> **Status:** 🚧 Phases 1–3 complete — sequential (T0), OpenMP (T1), and **naïve CUDA (T2)** tiers for BFS / PageRank / WCC, plus a sequential Louvain (T0) and a binary→call-graph extractor. All GPU results validated against the T0 oracle. Real CPU-scaling and GPU results below. T3 (optimized CUDA) is next on the [roadmap](#roadmap).

---

## Why this study

Two problems I work on produce graphs with the *same* pathological shape:

| Domain | Vertices | Edges | Why it is skewed |
|---|---|---|---|
| **Binary analysis** | functions / basic blocks | calls / branches | a `malloc` wrapper or dispatch table is called from everywhere; leaf functions from nowhere |
| **Network intrusion detection** | hosts / flows | connections | a scanner or server touches thousands of peers; most hosts touch a few |

Both are **power-law**. On a GPU, when one thread handles a degree-10⁶ hub while its warp-mates handle degree-1 leaves, 31 lanes idle — *load imbalance* and *warp divergence* collapse utilization. This study measures, tier by tier, how much each standard GPU technique recovers on exactly these graphs.

The synthetic RMAT graphs the engine ships with already exhibit this: at scale 18 the generated graph has **max degree ≈ 59,000 vs. average degree ≈ 32** — a 1,800× spread.

---

## Algorithms × implementation tiers

Following the tiered-benchmark structure of [4], every algorithm is implemented at each tier so a speedup is attributable to one technique. `✅` done · `🚧` in progress · `⬜` planned.

| Algorithm | T0 · Seq C++ | T1 · OpenMP | T2 · Naïve CUDA | T3 · Optimized CUDA | T4 · Advanced |
|---|:--:|:--:|:--:|:--:|:--:|
| **BFS** | ✅ | ✅ | ✅ | ⬜ | ⬜ |
| **PageRank** | ✅ | ✅ | ✅ | 🚧 | ⬜ |
| **Connected Components (WCC)** | ✅ | ✅ | ✅ | ⬜ | ⬜ |
| **Community Detection (Louvain)** | ✅ | ⬜ | ⬜ | ⬜ | — |

**What each tier isolates**

- **T0 Sequential C++** — correctness oracle. Every faster tier is diffed against it (`tests/`).
- **T1 OpenMP** — multi-core CPU baseline. *GPU speedups are reported against T1, never against a single core* — comparing a GPU to one CPU core is the classic way to inflate a result.
- **T2 Naïve CUDA** — one thread per vertex, CSR, global memory. Honest starting point.
- **T3 Optimized CUDA** — shared-memory frontiers, memory coalescing, warp primitives (`__shfl`, `__ballot`), kernel fusion.
- **T4 Advanced** — well-known load-balancing strategies for skewed degree (thread-/warp-/block-per-vertex bucketing, à la Gunrock/Merrill), applied and measured on security graphs. Documented as **established technique**, not a novel contribution.

Louvain is the hardest to parallelize on a GPU and is scoped last; it may stop at T2.

---

## Results

<a name="results"></a>

### Phase 2 — OpenMP (T1) thread scaling

Speedup of the OpenMP tier over the sequential (T0) baseline, on a power-law RMAT
graph (2²⁰ vertices, ~16.8M edges), Intel Core i7-11850H (8 physical cores / 16
threads), median of 10 runs:

![OpenMP thread-scaling](docs/img/scaling.png)

| Algorithm | seq (T0) | 1 thread | 2 | 4 | 8 | 16 | Best speedup |
|---|--:|--:|--:|--:|--:|--:|:--:|
| **WCC**      | 232.7 ms | 1.32× | 2.48× | 4.25× | 5.67× | 6.04× | **6.0×** |
| **PageRank** | 1939.5 ms | 0.48× | 0.95× | 1.73× | 2.62× | 2.96× | **3.0×** |
| **BFS**      | 169.5 ms | 0.57× | 1.01× | 1.53× | 2.14× | 2.53× | **2.5×** |

What the numbers say — and *why* is the point of the study:

- **WCC scales best (6×)** — label propagation has no per-vertex synchronization,
  so it is limited only by memory bandwidth.
- **BFS scales worst (2.5×)** — the atomic frontier claim and the per-level barrier
  serialize work; this is the imbalance the GPU tiers are designed to attack.
- **PageRank sits between (3×)** — bandwidth-bound gather, no atomics after switching
  to a pull kernel.
- **1 thread is *slower* than T0** for BFS/PageRank — honest parallel-overhead cost,
  reported rather than hidden.
- **8→16 threads barely moves** — the CPU has 8 physical cores; the rest is
  hyper-threading. Textbook, and exactly why GPUs are the next step.

*Regenerate: `powershell -ExecutionPolicy Bypass -File .\scripts\scaling.ps1` → `results/scaling.csv`.*

### Phase 3 — Naïve CUDA (T2) vs CPU

Naïve GPU tier (one thread per vertex, CSR in global memory, no shared memory or
warp primitives yet) on a large power-law graph (2²⁰ vertices, 33.5M edges).
Hardware: NVIDIA T600 (sm_75) vs. Intel Core i7-11850H. All GPU outputs match the
sequential oracle exactly (BFS, WCC) / within tolerance (PageRank).

![Naïve CUDA vs CPU](docs/img/cuda_t2.png)

| Algorithm | CPU seq | Naïve CUDA (T2) | Speedup |
|---|--:|--:|:--:|
| **WCC** | 200.9 ms | 90.5 ms | **2.22×** |
| **BFS** | 139.0 ms | 96.5 ms | **1.44×** |
| **PageRank** | 1834 ms | 4296 ms | **0.43×** |

The honest finding — and the reason this is a *study*:

- **WCC and BFS already win** at the naïve tier on a large graph: enough work per
  kernel to hide launch and transfer overhead. (On the smaller 2¹⁸ graph BFS was
  *slower* — the GPU only pays off once the graph is big enough, itself a result
  worth showing.)
- **Naïve PageRank loses (0.43×)** — not because of the algorithm, but because its
  two per-iteration reductions (dangling mass, L1 change) are single-address
  `atomicAdd`s hammered by all N threads. That contention serializes the GPU. This
  is the textbook naïve-reduction bottleneck, and it is exactly what the **T3**
  tier fixes (block-level parallel reductions + shared memory).

*This is the point of the tiered design: the tier that fails tells you which
optimization matters next.*

---

## Quick start

<a name="quick-start"></a>

### Requirements
- **Phase 1 (today):** GCC ≥ 11, CMake ≥ 3.18, OpenMP. **No GPU required.**
- **Phase 3+ (GPU tiers):** CUDA Toolkit ≥ 12.x, NVIDIA GPU (compute capability ≥ 7.0). Developed locally on an NVIDIA T600 (CC 7.5); Colab/Kaggle notebooks provided for larger-scale runs — see `notebooks/`.

### Build & test
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure     # correctness oracle
```

### Run
```bash
# BFS on a synthetic power-law graph (2^18 vertices, avg degree 16)
./build/run_bench --algo bfs --tier seq --rmat 18 --ef 16 --source 0

# OpenMP tier (set threads via OMP_NUM_THREADS)
OMP_NUM_THREADS=8 ./build/run_bench --algo pagerank --tier openmp --rmat 20 --ef 16

# PageRank / WCC / Louvain on a graph file
./build/run_bench --algo wcc     --tier seq --graph data/graphs/sample.edges
./build/run_bench --algo louvain --tier seq --graph data/graphs/sample.edges

# Turn a real binary into a call-graph, then analyze it
python tools/extract_callgraph.py --binary /bin/ls --out data/graphs/ls --stats
./build/run_bench --algo louvain --tier seq --graph data/graphs/ls.edges

# Full benchmark sweep -> results/benchmarks.csv
./scripts/run_benchmarks.sh results/benchmarks.csv
```
Each run prints one CSV row: `algo,tier,graph,vertices,edges,max_degree,avg_degree,median_ms,summary`.
On Windows use `build\Release\run_bench.exe` and set threads with `$env:OMP_NUM_THREADS=8`.

---

## Design

<a name="design"></a>

### Separation of concerns (the key architectural decision)
```
 binary / pcap ──► extractor ──► edge list ──► CSR ──► [ T0│T1│T2│T3│T4 ] ──► metrics
   (tools/,        (Phase ≥2)     (data/)    (core/graph)   engine (core/)     (results/)
    Phase ≥2)
```
The engine consumes **CSR only** — it never parses a binary or a packet. The same
kernels therefore run unchanged on a call-graph and on a network-flow graph,
which is exactly what lets one study span both security domains.

### Repository layout
```
.
├── core/
│   ├── graph/      # csr.{hpp,cpp}, generator.{hpp,cpp}  — CSR + RMAT
│   └── cpu/        # algorithms.{hpp,cpp}                — T0 seq (the oracle)
├── bench/          # timer.hpp                           — median-of-N harness
├── apps/           # run_bench.cpp                       — unified CLI driver
├── tests/          # test_correctness.cpp               — known-answer tests (CI)
├── scripts/        # run_benchmarks.sh                   — sweep -> CSV
├── data/           # graphs + provenance/licensing notes
├── results/        # committed CSVs (+ figures, later)
├── notebooks/      # Colab/Kaggle GPU runners            (Phase 3)
├── tools/          # binary/pcap -> graph extractors     (Phase ≥2)
└── docs/           # ROADMAP.md, design notes, paper draft
```

### What exists after Phase 1
- `CsrGraph` — CSR structure, edge-list loader, degree/stats, edge→CSR builder (with optional symmetrization for undirected algorithms).
- `make_rmat` — deterministic Graph500-parameter RMAT generator (reproducible from a seed); `make_test_graph` — a 7-vertex hand-checkable graph.
- `seq::bfs`, `seq::pagerank`, `seq::wcc` — the sequential reference tier.
- `time_median_ms` — the repeat-and-median timing harness.
- 19 known-answer assertions in `tests/`, all passing.

---

## Reproducibility

<a name="reproducibility"></a>

Every number is regenerated by `scripts/run_benchmarks.sh`; raw CSVs live in `results/`. Following [4]:

- **Hardware:** NVIDIA T600 Laptop GPU (4 GB, Turing, CC 7.5) · Intel Core i7-11850H (8 cores / 16 threads) · 32 GB RAM · Dell Precision 3561.
- **Software:** CUDA «12.x» · «MSVC / GCC» · `-O3` · driver «version» · Windows 11.
- **Protocol:** each configuration run **10×**, **median** reported; GPU end-to-end latency will be decomposed into **host→device transfer** vs **on-device kernel** time. GPU speedups are stated against the **OpenMP** baseline (T1) and include transfer time unless labeled kernel-only.
- **Correctness:** every non-T0 result is diffed against the T0 oracle in `tests/`; CI fails on any mismatch. RMAT graphs are deterministic given a seed.

---

## Roadmap

<a name="roadmap"></a>

- [x] **Phase 1 — Baseline & harness.** ✅ CSR core, RMAT generator, T0 BFS/PageRank/WCC, timing harness, correctness tests, CLI, benchmark sweep, this README.
- [x] **Phase 2 — Parallel CPU (T1).** ✅ OpenMP BFS/PageRank/WCC, validated against T0; thread-scaling study (chart above); PowerShell benchmark runner.
- [ ] **Phase 3 — GPU (T2).** Naïve CUDA for all three algorithms; Colab/Kaggle notebook; Nsight baseline profiles.
- [ ] **Phase 4 — Optimization (T3).** Shared memory, coalescing, warp primitives, kernel fusion.
- [ ] **Phase 5 — Advanced load balancing (T4).** Degree-bucketed scheduling; the GPU-utilization-recovery study on skewed security graphs.
- [ ] **Phase 6 — Scale & writeup.** Louvain; scale study to 100M+ edges; cross-dataset generalization on IDS data; `docs/` paper draft → arXiv preprint.

Detailed weekly milestones in [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## References

<a name="references"></a>

1. *From packets to predictions on GPU: Accelerated graph-based intrusion detection system.* Computer Networks, 2025. — the domain bridge: packets → graph → GPU.
2. Zhang et al. *POEGA: Proxy-guided Evolving Graph Analytics on GPUs.* OSDI 2026. — GPU graph analytics on billion-edge graphs; irregularity handling; geometric-mean speedup reporting.
3. *Graphite: A GPU-Accelerated Mixed-Precision Graph Optimization Framework.* arXiv:2509.26581. — mixed precision + load balancing + benchmark reporting conventions.
4. *GPU-Accelerated Algorithms for Graph Vector Search: Taxonomy, Empirical Study, and Research Directions.* arXiv:2602.16719. — the taxonomy / empirical-study template this repo follows.

## Citation
See `CITATION.cff`.

## License
«MIT» — see `LICENSE`. Extracted graphs retain the license of their source; see `data/README.md`.

## Author
**Emad Mahmodi** · [portfolio](https://emad-mahmodi.github.io)
