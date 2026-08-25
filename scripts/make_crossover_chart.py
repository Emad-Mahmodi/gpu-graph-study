#!/usr/bin/env python3
"""Crossover chart: when does the T3 frontier/active-set optimization beat the
T2 naive kernel? It depends on graph diameter.

Measured on NVIDIA T600 (sm_75) vs Intel i7-11850H, median of 10 runs.
  RMAT s20 : 2^20 vertices, 33.5M edges, BFS depth 4      (low diameter)
  Grid 1024: 2^20 vertices,  4.2M edges, BFS depth 2046   (high diameter)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# median_ms measured on the T600.
data = {
    "BFS": {"rmat": {"t2": 102.278, "t3": 138.559},
            "grid": {"t2": 200.092, "t3": 83.9267}},
    "WCC": {"rmat": {"t2": 88.6794, "t3": 96.3744},
            "grid": {"t2": 521.535, "t3": 397.896}},
}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8))

# --- Left: T3-over-T2 speedup ratio on each graph type -----------------------
algos = ["BFS", "WCC"]
rmat_ratio = [data[a]["rmat"]["t2"] / data[a]["rmat"]["t3"] for a in algos]
grid_ratio = [data[a]["grid"]["t2"] / data[a]["grid"]["t3"] for a in algos]

x = np.arange(len(algos))
w = 0.35
b1 = axL.bar(x - w/2, rmat_ratio, w, label="RMAT (low diameter, depth 4)",
             color="#c44e52", edgecolor="black", linewidth=0.6)
b2 = axL.bar(x + w/2, grid_ratio, w, label="Grid (high diameter, depth 2046)",
             color="#2e8b57", edgecolor="black", linewidth=0.6)
axL.axhline(1.0, color="#444", linestyle="--", linewidth=1.2)
axL.text(len(algos)-0.5, 1.03, "break-even (T3 = T2)", ha="right", va="bottom",
         fontsize=9, color="#444")
for bars in (b1, b2):
    for b in bars:
        h = b.get_height()
        axL.text(b.get_x()+b.get_width()/2, h+0.03, f"{h:.2f}×",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
axL.set_xticks(x); axL.set_xticklabels(algos)
axL.set_ylabel("T3 speedup over T2  (>1 = optimization pays off)")
axL.set_title("The frontier/active-set optimization pays off\nonly on high-diameter graphs")
axL.set_ylim(0, max(grid_ratio+rmat_ratio)*1.25)
axL.legend(fontsize=8, loc="upper left")
axL.grid(axis="y", alpha=0.3)

# --- Right: absolute times incl. CPU, to show the launch-bound regime --------
labels = ["BFS\nRMAT", "BFS\nGrid", "WCC\nRMAT", "WCC\nGrid"]
cpu = [153.109, 17.483, 218.655, 19.5131]
t2  = [102.278, 200.092, 88.6794, 521.535]
t3  = [138.559, 83.9267, 96.3744, 397.896]
x = np.arange(len(labels)); w = 0.26
axR.bar(x - w, cpu, w, label="CPU (seq)",       color="#4c72b0", edgecolor="black", linewidth=0.5)
axR.bar(x,     t2,  w, label="Naïve CUDA (T2)",  color="#d9822b", edgecolor="black", linewidth=0.5)
axR.bar(x + w, t3,  w, label="Optimized (T3)",   color="#2e8b57", edgecolor="black", linewidth=0.5)
axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=9)
axR.set_ylabel("Median wall-clock (ms, lower = better)")
axR.set_title("On high-diameter grids the GPU is launch-bound:\ntiny per-level work, thousands of kernel launches")
axR.legend(fontsize=8)
axR.grid(axis="y", alpha=0.3)

fig.suptitle("Traversal (BFS / WCC): the optimal GPU strategy depends on graph diameter  ·  T600 vs i7-11850H",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig("docs/img/crossover.png", dpi=130, bbox_inches="tight")
print("wrote docs/img/crossover.png")
for a in algos:
    print(f"{a}: RMAT T3/T2={data[a]['rmat']['t2']/data[a]['rmat']['t3']:.2f}x, "
          f"Grid T3/T2={data[a]['grid']['t2']/data[a]['grid']['t3']:.2f}x")
