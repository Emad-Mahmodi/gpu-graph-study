#!/usr/bin/env python3
"""Regenerate every results figure from ONE consistent dataset, so the README
tables and charts never disagree. All numbers are median-of-10 wall-clock (ms)
measured on NVIDIA T600 (sm_75, 4 GB) vs Intel i7-11850H (8c/16t), CUDA 12.8.

Tiers: T0 = sequential C++ ; T1 = OpenMP (8 threads) ; T2 = naive CUDA ;
       T3 = optimized CUDA.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- canonical dataset ------------------------------------------------------
# RMAT s20-ef16: 2^20 vertices, 33.5M edges, BFS depth 4  (low diameter)
RMAT = {
    "BFS":      {"T0": 153.109, "T1": 75.1992, "T2": 102.278, "T3": 138.559},
    "WCC":      {"T0": 218.655, "T1": 46.2037, "T2": 88.6794, "T3": 96.3744},
    "PageRank": {"T0": 1866.93, "T1": 744.683, "T2": 4386.55, "T3": 1153.61},
}
# Grid 1024: 2^20 vertices, 4.2M edges, BFS depth 2046  (high diameter)
GRID = {
    "BFS": {"T0": 17.483,  "T1": 36.9317, "T2": 200.092, "T3": 83.9267},
    "WCC": {"T0": 19.5131, "T1": 9.3099,  "T2": 521.535, "T3": 397.896},
}

COL = {"T0": "#8a8a8a", "T1": "#4c72b0", "T2": "#d9822b", "T3": "#2e8b57"}
NAME = {"T0": "CPU 1-core (T0)", "T1": "CPU 8-core (T1)",
        "T2": "Naïve CUDA (T2)", "T3": "Optimized CUDA (T3)"}


def dump_csv():
    import io
    lines = ["algo,graph,tier,median_ms"]
    for graph, d in (("rmat-s20-ef16", RMAT), ("grid-1024", GRID)):
        for algo, row in d.items():
            for tier, ms in row.items():
                lines.append(f"{algo},{graph},{tier},{ms}")
    with open("results/benchmarks.csv", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote results/benchmarks.csv")


# ---- Figure 1: naive GPU (T2) vs both CPU baselines, on RMAT ----------------
def fig_t2():
    algos = ["BFS", "WCC", "PageRank"]
    vs_t0 = [RMAT[a]["T0"] / RMAT[a]["T2"] for a in algos]
    vs_t1 = [RMAT[a]["T1"] / RMAT[a]["T2"] for a in algos]
    x = np.arange(len(algos)); w = 0.35
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    b1 = ax.bar(x - w/2, vs_t0, w, label="vs CPU 1-core (T0)", color="#9ecae1", edgecolor="black", lw=0.6)
    b2 = ax.bar(x + w/2, vs_t1, w, label="vs CPU 8-core (T1)", color="#08519c", edgecolor="black", lw=0.6)
    ax.axhline(1.0, color="#444", ls="--", lw=1.2)
    ax.text(len(algos)-0.5, 1.03, "break-even", ha="right", va="bottom", fontsize=9, color="#444")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.03, f"{b.get_height():.2f}×",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(algos)
    ax.set_ylabel("Naïve-CUDA (T2) speedup  (>1 = GPU faster)")
    ax.set_title("Phase 3 — naïve GPU beats a single CPU core (except PageRank),\nbut not the 8-core CPU  ·  RMAT 2²⁰ / 33.5M edges")
    ax.set_ylim(0, max(vs_t0+vs_t1)*1.25); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig("docs/img/cuda_t2.png", dpi=130, bbox_inches="tight")
    print("wrote docs/img/cuda_t2.png")


# ---- Figure 2: PageRank T0/T1/T2/T3 absolute times -------------------------
def fig_t3():
    tiers = ["T0", "T1", "T2", "T3"]
    times = [RMAT["PageRank"][t] for t in tiers]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.bar([NAME[t] for t in tiers], times, color=[COL[t] for t in tiers],
                  edgecolor="black", lw=0.6, width=0.62)
    for b, t in zip(bars, times):
        ax.text(b.get_x()+b.get_width()/2, t+70, f"{t:.0f} ms",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Median wall-clock (ms, lower = better)")
    ax.set_title("Phase 4 — Optimized CUDA (T3) PageRank\n"
                 "3.8× faster than naïve GPU · 1.62× vs 1 core · still behind the 8-core CPU")
    ax.set_ylim(0, max(times)*1.16); ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), fontsize=9)
    fig.tight_layout(); fig.savefig("docs/img/cuda_t3.png", dpi=130, bbox_inches="tight")
    print("wrote docs/img/cuda_t3.png")


# ---- Figure 3: diameter crossover ------------------------------------------
def fig_crossover():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8))
    algos = ["BFS", "WCC"]
    rmat_r = [RMAT[a]["T2"] / RMAT[a]["T3"] for a in algos]
    grid_r = [GRID[a]["T2"] / GRID[a]["T3"] for a in algos]
    x = np.arange(len(algos)); w = 0.35
    b1 = axL.bar(x - w/2, rmat_r, w, label="RMAT (low diameter, depth 4)", color="#c44e52", edgecolor="black", lw=0.6)
    b2 = axL.bar(x + w/2, grid_r, w, label="Grid (high diameter, depth 2046)", color="#2e8b57", edgecolor="black", lw=0.6)
    axL.axhline(1.0, color="#444", ls="--", lw=1.2)
    axL.text(len(algos)-0.5, 1.03, "break-even (T3 = T2)", ha="right", va="bottom", fontsize=9, color="#444")
    for bars in (b1, b2):
        for b in bars:
            axL.text(b.get_x()+b.get_width()/2, b.get_height()+0.03, f"{b.get_height():.2f}×",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    axL.set_xticks(x); axL.set_xticklabels(algos)
    axL.set_ylabel("T3 speedup over T2  (>1 = optimization pays off)")
    axL.set_title("The frontier/active-set optimization pays off\nonly on high-diameter graphs")
    axL.set_ylim(0, max(rmat_r+grid_r)*1.25); axL.legend(fontsize=8, loc="upper left"); axL.grid(axis="y", alpha=0.3)

    labels = ["BFS\nRMAT", "BFS\nGrid", "WCC\nRMAT", "WCC\nGrid"]
    keys = [(RMAT, "BFS"), (GRID, "BFS"), (RMAT, "WCC"), (GRID, "WCC")]
    series = {t: [d[a][t] for (d, a) in keys] for t in ("T0", "T1", "T2", "T3")}
    x = np.arange(len(labels)); w = 0.2
    for i, t in enumerate(("T0", "T1", "T2", "T3")):
        axR.bar(x + (i-1.5)*w, series[t], w, label=NAME[t], color=COL[t], edgecolor="black", lw=0.4)
    axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=9)
    axR.set_ylabel("Median wall-clock (ms, lower = better)")
    axR.set_title("On high-diameter grids the GPU is launch-bound;\nthe 8-core CPU (T1) wins across the board")
    axR.legend(fontsize=7.5); axR.grid(axis="y", alpha=0.3)

    fig.suptitle("Traversal (BFS / WCC): optimal strategy depends on graph diameter  ·  T600 vs i7-11850H",
                 fontsize=11, y=1.02)
    fig.tight_layout(); fig.savefig("docs/img/crossover.png", dpi=130, bbox_inches="tight")
    print("wrote docs/img/crossover.png")


if __name__ == "__main__":
    dump_csv(); fig_t2(); fig_t3(); fig_crossover()
    # handy ratios
    for a in ("BFS", "WCC", "PageRank"):
        r = RMAT[a]
        print(f"RMAT {a}: T2 vs T1={r['T1']/r['T2']:.2f}x, T3 vs T1={r['T1']/r['T3']:.2f}x")
