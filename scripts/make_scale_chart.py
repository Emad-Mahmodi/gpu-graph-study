#!/usr/bin/env python3
"""Phase 6 scale study: does the CPU's win at 2^20 survive a bigger graph?

Every earlier result in this repo was measured at one size (2^20 vertices,
33.5M edges), where a strong 8-core i7 beat an entry-level T600 almost
everywhere. That leaves the interesting question unanswered: was that a fact
about the *hardware*, or about the *problem size*? A GPU has fixed costs --
host-to-device transfer, kernel launches, per-round host syncs -- that a small
graph cannot amortize.

This chart answers it from results/scale_study.csv (RMAT 2^18..2^23, every tier
at every size, one session). Top row: wall-clock against edge count, log-log,
so a straight line means linear scaling. Bottom: the ratio that matters --
best GPU tier over the full 8-core CPU -- with the break-even line at 1.0.
"""
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIERS = ["T0", "T1", "T2", "T3", "T4"]
TIER_LABEL = {
    "T0": "CPU 1-core (T0)",
    "T1": "CPU 8-core (T1)",
    "T2": "Naive CUDA (T2)",
    "T3": "Optimized CUDA (T3)",
    "T4": "Load-balanced CUDA (T4)",
}
TIER_STYLE = {
    "T0": dict(color="#4c72b0", marker="o", linestyle="--"),
    "T1": dict(color="#1f3f6e", marker="s", linestyle="-"),
    "T2": dict(color="#d9822b", marker="^", linestyle=":"),
    "T3": dict(color="#8c8c8c", marker="v", linestyle="-."),
    "T4": dict(color="#2e8b57", marker="D", linestyle="-"),
}
ALGOS = ["bfs", "pagerank", "wcc"]
ALGO_TITLE = {"bfs": "BFS", "pagerank": "PageRank", "wcc": "WCC"}

# data[algo][tier] -> list of (edges, ms), ordered by scale
data = defaultdict(lambda: defaultdict(list))
edges_of_scale = {}
with open("results/scale_study.csv", encoding="utf-8-sig", newline="") as fh:
    for row in csv.DictReader(fh):
        scale = int(row["scale"])
        edges = int(row["edges"])
        edges_of_scale[scale] = edges
        data[row["algo"]][row["tier"]].append((scale, edges, float(row["median_ms"])))

for algo in data:
    for tier in data[algo]:
        data[algo][tier].sort()

scales = sorted(edges_of_scale)

fig = plt.figure(figsize=(15, 8.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.42, wspace=0.24)

# --- Top row: scalability curves, one per algorithm -------------------------
for i, algo in enumerate(ALGOS):
    ax = fig.add_subplot(gs[0, i])
    for tier in TIERS:
        pts = data[algo].get(tier, [])
        if not pts:
            continue
        xs = [e for _, e, _ in pts]
        ys = [ms for _, _, ms in pts]
        ax.plot(xs, ys, label=TIER_LABEL[tier], linewidth=1.7,
                markersize=5, **TIER_STYLE[tier])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("directed edges (log)")
    if i == 0:
        ax.set_ylabel("median wall-clock, ms (log)")
    ax.set_title(ALGO_TITLE[algo])
    ax.grid(alpha=0.3, which="both")
    if i == 0:
        ax.legend(fontsize=7.5, loc="upper left")

# --- Bottom: the crossover -- best GPU tier vs the full 8-core CPU ----------
axc = fig.add_subplot(gs[1, :])
crossings = {}
for algo in ALGOS:
    t1 = {s: ms for s, _, ms in data[algo]["T1"]}
    t4 = {s: ms for s, _, ms in data[algo]["T4"]}
    xs = [edges_of_scale[s] for s in scales if s in t1 and s in t4]
    ys = [t1[s] / t4[s] for s in scales if s in t1 and s in t4]
    axc.plot(xs, ys, marker="o", linewidth=2, markersize=6, label=ALGO_TITLE[algo])
    for x, y in zip(xs, ys):
        axc.annotate("{:.2f}x".format(y), (x, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=7.5)
    # First size at which the GPU overtakes the CPU.
    for s in scales:
        if s in t1 and s in t4 and t1[s] / t4[s] >= 1.0:
            crossings[algo] = s
            break

axc.axhline(1.0, color="#444", linestyle="--", linewidth=1.3)
axc.text(max(edges_of_scale.values()), 0.97, "break-even: T4 GPU = 8-core CPU",
         fontsize=8.5, color="#444", va="top", ha="right")
axc.set_xscale("log")
axc.set_xlabel("directed edges (log)")
axc.set_ylabel("T4 speedup over the 8-core CPU")
axc.set_title("The CPU's advantage is a property of problem size, not of the hardware: "
              "the GPU overtakes it as the graph grows")
axc.grid(alpha=0.3, which="both")
axc.legend(fontsize=9)

fig.suptitle("Scale study -- RMAT 2^18 to 2^23 (0.5M to 268M directed edges)"
             "   |   NVIDIA T600 (4 GB) vs Intel i7-11850H (8 cores)",
             fontsize=11.5, y=0.975)
fig.savefig("docs/img/scale_study.png", dpi=130, bbox_inches="tight")
print("wrote docs/img/scale_study.png")

for algo in ALGOS:
    t1 = {s: ms for s, _, ms in data[algo]["T1"]}
    t4 = {s: ms for s, _, ms in data[algo]["T4"]}
    line = []
    for s in scales:
        if s in t1 and s in t4:
            line.append("s{}={:.2f}x".format(s, t1[s] / t4[s]))
    cross = crossings.get(algo)
    note = ("crosses at 2^{} ({:,} edges)".format(cross, edges_of_scale[cross])
            if cross else "no crossover in this range")
    print("{:9s} T4 vs 8-core CPU: {}  -> {}".format(algo, "  ".join(line), note))
