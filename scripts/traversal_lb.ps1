# scripts/traversal_lb.ps1
# Phase 5 sweep for the traversal algorithms: BFS and WCC across every tier
# (T0 seq, T1 OpenMP-8, T2 naive CUDA, T3 optimized CUDA, T4 degree-bucketed),
# on both graph regimes the study contrasts:
#   * RMAT  - power-law, shallow (BFS depth ~4)     -> huge frontiers
#   * grid  - uniform degree, deep (BFS depth 2046) -> tiny frontiers
#
# All five tiers are measured in one run so the numbers are mutually consistent
# (the repo's rule: no table mixes figures from different sessions).
# Writes results/traversal_lb.csv.
#
# Usage (from the repo root, after building):
#   powershell -ExecutionPolicy Bypass -File .\scripts\traversal_lb.ps1

param(
    [int]$Scale = 20,
    [int]$EdgeFactor = 16,
    [int]$GridSide = 1024,
    [int]$Threads = 8,
    [string]$Bin = ".\build\Release\run_bench.exe",
    [string]$Out = "results\traversal_lb.csv"
)

if (-not (Test-Path $Bin)) { $Bin = ".\build\run_bench.exe" }
if (-not (Test-Path $Bin)) { $Bin = ".\build\run_bench" }
if (-not (Test-Path $Bin)) { Write-Error "run_bench not found. Build first (see README)."; exit 1 }

New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null
"algo,graph,tier,median_ms,summary" | Out-File -Encoding utf8 $Out

# tier label -> --tier flag
$tiers = [ordered]@{ "T0" = "seq"; "T1" = "openmp"; "T2" = "cuda"; "T3" = "cuda_opt"; "T4" = "cuda_lb" }

# graph label -> the run_bench flags that select it
$graphs = [ordered]@{
    ("rmat-s{0}-ef{1}" -f $Scale, $EdgeFactor) = @("--rmat", "$Scale", "--ef", "$EdgeFactor")
    ("grid-{0}"        -f $GridSide)           = @("--grid", "$GridSide")
}

foreach ($algo in @("bfs", "wcc")) {
    foreach ($graph in $graphs.Keys) {
        Write-Host ("== {0} on {1} ==" -f $algo, $graph) -ForegroundColor Cyan
        $baseMs = 0.0
        foreach ($tier in $tiers.Keys) {
            if ($tiers[$tier] -eq "openmp") { $env:OMP_NUM_THREADS = $Threads }
            $row = & $Bin --algo $algo --tier $tiers[$tier] @($graphs[$graph])
            Remove-Item Env:\OMP_NUM_THREADS -ErrorAction SilentlyContinue
            if ($LASTEXITCODE -ne 0) { Write-Error "run failed: $algo $tier $graph"; exit 1 }
            $f  = $row -split ','
            $ms = [double]$f[7]
            $summary = $f[8]
            # T2 is the reference the GPU optimizations are attributed against.
            if ($tier -eq "T2") { $baseMs = $ms }
            ("{0},{1},{2},{3},{4}" -f $algo, $graph, $tier, $ms, $summary) |
                Out-File -Append -Encoding utf8 $Out
            $vsT2 = ""
            if ($baseMs -gt 0 -and $tier -match "T3|T4") { $vsT2 = "{0:N2}x vs T2" -f ($baseMs / $ms) }
            Write-Host ("  {0}  {1} ms   {2}" -f $tier, ("{0:N1}" -f $ms).PadLeft(9), $vsT2)
        }
        Write-Host ""
    }
}

Write-Host ("Wrote " + $Out) -ForegroundColor Green
