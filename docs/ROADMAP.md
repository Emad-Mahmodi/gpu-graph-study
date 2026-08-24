# Detailed roadmap

Weekly-milestone breakdown. Each phase ends with something that builds, passes
tests, and produces a benchmark row — never a half-finished tier.

## Phase 1 — Baseline & harness ✅ (done)
- [x] CSR data structure + edge-list loader + stats
- [x] RMAT generator (deterministic) + tiny known-answer test graph
- [x] T0 sequential BFS, PageRank, WCC (the correctness oracle)
- [x] Median-of-N timing harness
- [x] Unified CLI driver (`run_bench`)
- [x] 19 known-answer correctness assertions (all pass)
- [x] Benchmark sweep script → CSV
- [x] CMake build, `.gitignore`, README, data provenance notes

## Phase 2 — Parallel CPU (T1 · OpenMP) — ~1 week
- [ ] OpenMP BFS (frontier-based, per-level parallel expand)
- [ ] OpenMP PageRank (parallel edge/vertex loops, atomic or push/pull)
- [ ] OpenMP WCC (parallel label propagation or union-find)
- [ ] Diff every T1 result against T0 in `tests/`
- [ ] First speedup table: T1 vs T0, thread-scaling plot (1→N threads)
- [ ] `tools/extract_callgraph.py` first cut (angr headless) → real graph in `data/`
- **Milestone:** speedup-vs-threads figure + one real binary graph loaded end-to-end.

## Phase 3 — GPU baseline (T2 · Naïve CUDA) — ~1–2 weeks
- [ ] `enable_language(CUDA)` in CMake; CSR upload to device
- [ ] Naïve one-thread-per-vertex BFS, PageRank, WCC kernels (global memory)
- [ ] Host↔device transfer timing separated from kernel timing
- [ ] `notebooks/run_on_colab.ipynb` (clone + build + run on Colab/Kaggle T4)
- [ ] Nsight Compute baseline profiles (occupancy, warp efficiency) committed
- **Milestone:** first honest T2-vs-T1 speedup + a profile showing the imbalance.

## Phase 4 — Optimization (T3) — ~2 weeks
- [ ] Memory coalescing on CSR access
- [ ] Shared-memory frontier / tiling
- [ ] Warp-level primitives (`__shfl_*`, `__ballot_sync`) for reductions
- [ ] Kernel fusion where it helps (e.g. PageRank iterations)
- [ ] Ablation: turn each optimization on/off, attribute the speedup
- **Milestone:** per-technique ablation table.

## Phase 5 — Advanced load balancing (T4) — ~2 weeks
- [ ] Degree bucketing: thread-per-vertex (low) / warp-per-vertex (mid) / block-per-vertex (high)
- [ ] Runtime utilization monitor; re-bucket between BFS levels
- [ ] The headline study: GPU-utilization recovery on the most skewed graphs
- [ ] Clearly documented as established technique (Gunrock/Merrill), measured here
- **Milestone:** utilization-recovery figure across all datasets.

## Phase 6 — Scale, extra algorithm, writeup — ~2–3 weeks
- [ ] Louvain community detection (T0→T2 at least)
- [ ] Scale study: RMAT to 100M+ edges; scalability curves
- [ ] Cross-dataset generalization on IDS graphs (CICIDS→UNSW), mirroring ref [1]
- [ ] `docs/paper/` draft; figures auto-generated from `results/`
- **Milestone:** arXiv-ready preprint + a repo any reviewer can reproduce.

## Ground rules (kept every phase)
1. Nothing merges unless it builds and `ctest` is green.
2. Every GPU result is diffed against the T0 oracle.
3. Speedups are vs. the OpenMP baseline, geometric mean over datasets, median of 10 runs.
4. No graph from proprietary/NDA firmware is ever committed.
