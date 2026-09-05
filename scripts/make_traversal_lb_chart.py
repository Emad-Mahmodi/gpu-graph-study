#!/usr/bin/env python3
"""Phase 5 traversal chart: what does degree-bucketing (T4) add on top of the
T3 frontier/active-set kernels, and where does it cost instead of pay?

T4 = T3 + Gunrock/Merrill degree-bucketing, so the T4/T3 ratio isolates exactly
one technique. The answer turns on how *skewed* the degree distribution is —
a different axis than the T3 study's diameter crossover:

  RMAT s20 : 2^20 vertices, 33.5M edges, max degree 138,626 vs avg 32 (skewed)
  Grid 1024: 2^20 vertices,  4.2M edges, every degree <= 4        (uniform)

Measured on NVIDIA T600 (sm_75) vs Intel i7-11850H, median of 10 runs, all five
tiers in one session (results/traversal_lb.csv).
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Read the sweep so the figure can never drift from the committed numbers.
data = {}
with open("results/traversal_lb.csv", encoding="utf-8-sig", newline="") as fh:
    for row in csv.DictReader(fh):
        kind = "rmat" if row["graph"].startswith("rmat") else "grid"
        data.setdefault(row["algo"].upper(), {}).setdefault(kind, {})[row["tier"]] = \
            float(row["median_ms"])

cases = [("BFS", "rmat"), ("BFS", "grid"), ("WCC", "rmat"), ("WCC", "grid")]
labels = ["BFS\nRMAT", "BFS\nGrid", "WCC\nRMAT", "WCC\nGrid"]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8))

# --- Left: T4-over-T3 ratio = the bucketing technique, isolated --------------
algos = ["BFS", "WCC"]
skewed  = [data[a]["rmat"]["T3"] / data[a]["rmat"]["T4"] for a in algos]
uniform = [data[a]["grid"]["T3"] / data[a]["grid"]["T4"] for a in algos]

x = np.arange(len(algos))
w = 0.35
b1 = axL.bar(x - w/2, skewed, w, label="RMAT (power-law: max deg 138,626 vs avg 32)",
             color="#2e8b57", edgecolor="black", linewidth=0.6)
b2 = axL.bar(x + w/2, uniform, w, label="Grid (uniform: every degree ≤ 4)",
             color="#c44e52", edgecolor="black", linewidth=0.6)
axL.axhline(1.0, color="#444", linestyle="--", linewidth=1.2)
axL.text(len(algos)-0.5, 1.03, "break-even (T4 = T3)", ha="right", va="bottom",
         fontsize=9, color="#444")
for bars in (b1, b2):
    for b in bars:
        h = b.get_height()
        axL.text(b.get_x()+b.get_width()/2, h+0.04, f"{h:.2f}×",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
axL.set_xticks(x); axL.set_xticklabels(algos)
axL.set_ylabel("T4 speedup over T3  (>1 = bucketing pays off)")
axL.set_title("Degree-bucketing pays off in proportion to degree skew —\n"
              "on a uniform-degree graph it is pure overhead")
axL.set_ylim(0, max(skewed + uniform) * 1.28)
axL.legend(fontsize=8, loc="upper right")
axL.grid(axis="y", alpha=0.3)

# --- Right: absolute times against the *best* CPU baseline ------------------
cpu8 = [data[a][k]["T1"] for a, k in cases]
t2   = [data[a][k]["T2"] for a, k in cases]
t3   = [data[a][k]["T3"] for a, k in cases]
t4   = [data[a][k]["T4"] for a, k in cases]

x = np.arange(len(labels)); w = 0.2
axR.bar(x - 1.5*w, cpu8, w, label="CPU 8-core (T1)",   color="#4c72b0", edgecolor="black", linewidth=0.5)
axR.bar(x - 0.5*w, t2,   w, label="Naïve CUDA (T2)",   color="#d9822b", edgecolor="black", linewidth=0.5)
axR.bar(x + 0.5*w, t3,   w, label="Optimized (T3)",    color="#8c8c8c", edgecolor="black", linewidth=0.5)
axR.bar(x + 1.5*w, t4,   w, label="Load-balanced (T4)",color="#2e8b57", edgecolor="black", linewidth=0.5)

# Mark the one case where a GPU tier finally beats the full multi-core CPU.
axR.annotate("first GPU tier to beat\nthe 8-core CPU",
             xy=(0 + 1.5*w, t4[0]), xytext=(0.35, max(t4[0], cpu8[0]) * 3.2),
             fontsize=8.5, ha="left", color="#1a5c37",
             arrowprops=dict(arrowstyle="->", color="#1a5c37", linewidth=1.1))

axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=9)
axR.set_ylabel("Median wall-clock (ms, lower = better)")
axR.set_title("Absolute times against the best CPU baseline\n(the grid stays launch-bound at every GPU tier)")
axR.legend(fontsize=8)
axR.grid(axis="y", alpha=0.3)

fig.suptitle("Traversal T4 — degree-bucketed BFS / WCC  ·  T600 vs i7-11850H (8 cores)",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig("docs/img/cuda_t4_traversal.png", dpi=130, bbox_inches="tight")
print("wrote docs/img/cuda_t4_traversal.png")
for a, k in cases:
    d = data[a][k]
    print(f"{a:4s} {k:4s}  T2={d['T2']:8.1f}  T3={d['T3']:8.1f}  T4={d['T4']:8.1f}  "
          f"T4/T3={d['T3']/d['T4']:.2f}x  T4/T2={d['T2']/d['T4']:.2f}x  "
          f"T4 vs CPU8={d['T1']/d['T4']:.2f}x")
