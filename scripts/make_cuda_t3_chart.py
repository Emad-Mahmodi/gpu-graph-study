#!/usr/bin/env python3
"""Phase 4 (T3) PageRank chart: naive CUDA (T2) vs optimized CUDA (T3) vs CPU.
Measured on NVIDIA T600 (sm_75) vs Intel i7-11850H, RMAT s20 (2^20 vertices,
33.5M edges), median of 10 runs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Measured median_ms.
cpu_seq = 1866.93
t2_naive = 4386.55
t3_opt = 1153.61

labels = ["Naïve CUDA (T2)", "Optimized CUDA (T3)"]
speedups = [cpu_seq / t2_naive, cpu_seq / t3_opt]   # vs CPU
colors = ["#d9822b" if s < 1 else "#2e8b57" for s in speedups]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

# --- Left: speedup over CPU -------------------------------------------------
bars = ax1.bar(labels, speedups, color=colors, width=0.55, edgecolor="black", linewidth=0.6)
ax1.axhline(1.0, color="#444", linestyle="--", linewidth=1.2)
ax1.text(1.48, 1.02, "CPU baseline (1.0×)", ha="right", va="bottom",
         fontsize=9, color="#444")
for b, s in zip(bars, speedups):
    ax1.text(b.get_x() + b.get_width() / 2, s + 0.03, f"{s:.2f}×",
             ha="center", va="bottom", fontsize=12, fontweight="bold")
ax1.set_ylabel("Speedup over CPU (higher = better)")
ax1.set_title("PageRank: naïve GPU loses, optimized GPU wins")
ax1.set_ylim(0, max(speedups) * 1.25)
ax1.grid(axis="y", alpha=0.3)

# --- Right: absolute time ---------------------------------------------------
tlabels = ["CPU\n(seq)", "Naïve\nCUDA (T2)", "Optimized\nCUDA (T3)"]
times = [cpu_seq, t2_naive, t3_opt]
tcolors = ["#4c72b0", "#d9822b", "#2e8b57"]
tbars = ax2.bar(tlabels, times, color=tcolors, width=0.6, edgecolor="black", linewidth=0.6)
for b, t in zip(tbars, times):
    ax2.text(b.get_x() + b.get_width() / 2, t + 60, f"{t:.0f} ms",
             ha="center", va="bottom", fontsize=11, fontweight="bold")
ax2.set_ylabel("Median wall-clock (ms, lower = better)")
ax2.set_title("Same result, 3.8× faster than naïve GPU")
ax2.set_ylim(0, max(times) * 1.18)
ax2.grid(axis="y", alpha=0.3)

fig.suptitle("Phase 4 — Optimized CUDA (T3) PageRank  ·  RMAT 2²⁰ / 33.5M edges  ·  NVIDIA T600 vs i7-11850H",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig("docs/img/cuda_t3.png", dpi=130, bbox_inches="tight")
print("wrote docs/img/cuda_t3.png")
print(f"T3 vs CPU:   {cpu_seq/t3_opt:.2f}x")
print(f"T3 vs naive: {t2_naive/t3_opt:.2f}x")
