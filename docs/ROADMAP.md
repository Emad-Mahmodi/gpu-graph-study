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

## Phase 2 — Parallel CPU (T1 · OpenMP) ✅ (done)
- [x] OpenMP BFS (frontier-based, per-level parallel expand)
- [x] OpenMP PageRank (parallel edge/vertex loops, atomic or push/pull)
- [x] OpenMP WCC (parallel label propagation or union-find)
- [x] Diff every T1 result against T0 in `tests/`
- [x] First speedup table: T1 vs T0, thread-scaling plot (1→N threads)
- [x] `tools/extract_callgraph.py` first cut → real graph in `data/` (`ls`, `notepad`)
- **Milestone:** ✅ speedup-vs-threads figure + one real binary graph loaded end-to-end.
- **Result:** WCC 6.0×, PageRank 3.0×, BFS 2.5× on 8 cores; 8→16 threads barely
  moves (hyper-threading), and 1 thread is *slower* than T0 for BFS/PageRank —
  the parallel-overhead cost, reported rather than hidden.

## Phase 3 — GPU baseline (T2 · Naïve CUDA) ✅ (done)
- [x] `enable_language(CUDA)` in CMake; CSR upload to device
- [x] Naïve one-thread-per-vertex BFS, PageRank, WCC kernels (global memory)
- [x] Every T2 result diffed against the T0 oracle (now a CI test)
- [ ] `notebooks/run_on_colab.ipynb` (clone + build + run on Colab/Kaggle T4) — Phase 6
- [ ] Nsight Compute profiles committed — deferred; the tier-to-tier timings
      already attribute each bottleneck, and Nsight adds a tool dependency a
      reproducer would have to install
- **Milestone:** ✅ first honest T2-vs-T1 comparison.
- **Result:** the naïve GPU beats one CPU core for WCC (2.5×) and BFS (1.5×) but
  loses to all 8 cores everywhere; naïve PageRank loses to *everything* (0.43×),
  and the measurement names the culprit — two single-address `atomicAdd`
  reductions per iteration. That is what Phase 4 removes.

## Phase 4 — Optimization (T3) ✅ (done)
- [x] Block-level shared-memory reductions (dangling mass + L1 change)
- [x] Precomputed inverse out-degree (removes a `double` divide from the hot loop)
- [x] Explicit frontier worklist for BFS; active-set label propagation for WCC
- [x] Warp-level primitives (`__shfl_down_sync`) for reductions — landed in T4
- [x] Ablation: each speedup attributed to one named technique
- **Milestone:** ✅ per-technique table + the diameter-crossover study.
- **Result:** PageRank 4.0× over naïve GPU (0.48× → 1.90× vs a CPU core), purely
  from the reduction fix. For traversal the verdict *flips with graph shape*:
  the frontier/active-set optimization loses on shallow RMAT (0.76× / 0.93×)
  and wins 2.47× on a high-diameter grid — where the GPU is nonetheless
  launch-bound and loses to a single CPU core outright.

## Phase 5 — Advanced load balancing (T4) ✅ (done)
- [x] Degree bucketing: thread-per-vertex (low ≤ 32) / warp-per-vertex (33–1024) /
      block-per-vertex (> 1024), for PageRank, BFS and WCC
- [x] Re-bucket between BFS levels — the frontier changes shape every level, so
      BFS re-partitions it *on the device* each round (PageRank and WCC can
      bucket once on the host, because their candidate set is all N)
- [x] The headline study: recovery on the most skewed graphs
- [x] Clearly documented as established technique (Gunrock/Merrill), measured here
- **Milestone:** ✅ recovery figures (`docs/img/cuda_t4.png`,
  `docs/img/cuda_t4_traversal.png`) across both graph regimes.
- **Result:** PageRank 5.16× over naïve GPU (within 15% of 8 cores); BFS 2.62×
  over T3 and **1.38× faster than the 8-core CPU** — the study's first and only
  GPU win against the full CPU; WCC 1.26× over T3. Two negative results kept
  because they are the point of a tiered study: the block-per-hub level gives
  *no* gain over warp-per-vertex on RMAT (too few vertices above degree 1024),
  and on a uniform-degree grid bucketing is a net **loss** (0.54× for BFS) —
  the mirror image of T3's diameter crossover, on the orthogonal axis of skew.

## Phase 6 — Scale, extra algorithm, writeup (in progress)
- [x] Scale study: RMAT 2¹⁸ → 2²³ (8.4M → 268M directed edges), every tier at
      every size in one session; scalability curves in `docs/img/scale_study.png`
- [x] Louvain at T1 (OpenMP), with the chunk-count quality/speed ablation and a
      thread-count-independence test
- [x] Louvain at T2 (naïve CUDA). Sort-based rather than hash-table-based on
      purpose: a hash table needs an overflow path for extreme hubs, which is
      most of the T4 machinery, and building it here would make the later
      speedup unattributable. Beats the 8-core CPU from 2¹⁶ upward (9.31× a
      single core at 2¹⁸) at ~73% of the oracle's modularity.
- [ ] Louvain T3: restore the incremental visibility that took T1 from 72.6% to
      93.6% quality, without giving up the GPU's parallelism
- [ ] Louvain T4: degree-bucketed accumulator sizing. Note this was originally
      marked "not applicable" — wrongly. Louvain needs a per-vertex accumulator
      over *distinct neighbouring communities*, whose size is set by degree, so
      bucketing is the precondition for using shared memory here rather than an
      optimization on top.
- [ ] Cross-dataset generalization on IDS graphs (CICIDS→UNSW), mirroring ref [1]
- [ ] `docs/paper/` draft; figures auto-generated from `results/`
- **Milestone:** arXiv-ready preprint + a repo any reviewer can reproduce.
- **Result so far:** the scale study answers the question the earlier phases
  could not. At the single size this repo used to report (33.5M edges) the
  8-core CPU won almost everything; by 268M edges PageRank T4 is **2.00×** the
  8-core CPU, BFS **1.58×**, and WCC has climbed from 0.47× to exactly
  break-even. Nothing in the kernels changed — only the graph grew, so the old
  verdict was a statement about problem size, not about the hardware. Two
  results worth keeping: **naïve CUDA (T2) never justifies itself** at any size
  tested (it only reaches the CPU at 268M, where the optimized tiers are already
  at 1.6–2.0×), and WCC's crossover sits just past what a 4 GB card can hold.
  Louvain is the one algorithm whose parallel tiers cannot reproduce the oracle.
  T1 trades ~6% of modularity for 3.6–6.5×. T2 is faster still — 9.31× a single
  core, beating the 8-core CPU from 2¹⁶ up — but at ~73% of the oracle's
  modularity, and the two numbers are never quoted apart. The 73% is the same
  figure the CPU chunk ablation measured for pure simultaneous updates (72.6%),
  reproduced by unrelated code on different hardware: the loss belongs to the
  formulation, not to an implementation.

## Ground rules (kept every phase)
1. Nothing merges unless it builds and `ctest` is green.
2. Every GPU result is diffed against the T0 oracle.
3. Speedups are vs. the OpenMP baseline, geometric mean over datasets, median of 10 runs.
4. No graph from proprietary/NDA firmware is ever committed.
