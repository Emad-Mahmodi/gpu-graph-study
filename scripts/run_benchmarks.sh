#!/usr/bin/env bash
# scripts/run_benchmarks.sh
# Runs the current-tier benchmark sweep and writes a CSV to results/.
# Extend the ALGOS / GRAPHS / TIERS arrays as later phases land.
set -euo pipefail

BIN=${BIN:-./build/run_bench}
OUT=${1:-results/benchmarks.csv}
mkdir -p "$(dirname "$OUT")"

ALGOS=(bfs pagerank wcc)   # louvain runs seq-only; add it below if desired
TIERS=(seq openmp)         # add: cuda ... in later phases
# RMAT graphs: "scale edge_factor"
RMATS=("14 16" "16 16" "18 8" "18 16")

echo "algo,tier,graph,vertices,edges,max_degree,avg_degree,median_ms,summary" > "$OUT"
for tier in "${TIERS[@]}"; do
  for algo in "${ALGOS[@]}"; do
    for r in "${RMATS[@]}"; do
      read -r scale ef <<< "$r"
      echo ">> $algo / $tier / rmat s$scale ef$ef" >&2
      "$BIN" --algo "$algo" --tier "$tier" --rmat "$scale" --ef "$ef" >> "$OUT"
    done
  done
done
echo "Wrote $OUT" >&2
