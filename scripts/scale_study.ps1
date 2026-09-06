# scripts/scale_study.ps1
# Phase 6 scale study: how does each tier behave as the graph grows?
#
# The Phase 3-5 results were all measured at one size (2^20 vertices, 33.5M
# edges), where a strong 8-core CPU beat the GPU almost everywhere. The open
# question the roadmap left is whether that verdict is a property of the
# *hardware* or of the *problem size* - a GPU needs enough work to amortize its
# transfers and launches. This sweeps RMAT from 2^18 to 2^23 (0.5M to 268M
# directed edges) with every tier at every size, so the curves answer it.
#
# Repeats shrink as the graph grows: the run-to-run spread also shrinks with
# size, and 10 repeats of a 30-second kernel is not a good use of an hour.
#
# Expect roughly 30-40 minutes end to end; most of it is RMAT generation, which
# run_bench redoes per invocation (deterministic given the seed, so every tier
# still sees an identical graph).
#
# Usage (from the repo root, after building):
#   powershell -ExecutionPolicy Bypass -File .\scripts\scale_study.ps1

param(
    [int[]]$Scales = @(18, 19, 20, 21, 22, 23),
    [int]$EdgeFactor = 16,
    [int]$Threads = 8,
    [string]$Bin = ".\build\Release\run_bench.exe",
    [string]$Out = "results\scale_study.csv"
)

if (-not (Test-Path $Bin)) { $Bin = ".\build\run_bench.exe" }
if (-not (Test-Path $Bin)) { $Bin = ".\build\run_bench" }
if (-not (Test-Path $Bin)) { Write-Error "run_bench not found. Build first (see README)."; exit 1 }

New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null
"algo,scale,vertices,edges,max_degree,tier,repeats,median_ms,summary" | Out-File -Encoding utf8 $Out

$tiers = [ordered]@{ "T0" = "seq"; "T1" = "openmp"; "T2" = "cuda"; "T3" = "cuda_opt"; "T4" = "cuda_lb" }

function Get-Repeats([int]$scale) {
    if ($scale -le 20) { return 10 }
    if ($scale -le 22) { return 5 }
    return 3
}

$started = Get-Date
foreach ($scale in $Scales) {
    $reps = Get-Repeats $scale
    Write-Host ("=== RMAT scale {0} (ef {1}), {2} repeats ===" -f $scale, $EdgeFactor, $reps) -ForegroundColor Cyan
    foreach ($algo in @("bfs", "pagerank", "wcc")) {
        $ref = 0.0
        foreach ($tier in $tiers.Keys) {
            if ($tiers[$tier] -eq "openmp") { $env:OMP_NUM_THREADS = $Threads }
            $row = & $Bin --algo $algo --tier $tiers[$tier] --rmat $scale --ef $EdgeFactor --repeats $reps
            Remove-Item Env:\OMP_NUM_THREADS -ErrorAction SilentlyContinue
            if ($LASTEXITCODE -ne 0) { Write-Error "run failed: $algo $tier scale $scale"; exit 1 }
            $f = $row -split ','
            # run_bench row: algo,tier,graph,vertices,edges,max_degree,avg_degree,median_ms,summary
            $ms = [double]$f[7]
            ("{0},{1},{2},{3},{4},{5},{6},{7},{8}" -f `
                $algo, $scale, $f[3], $f[4], $f[5], $tier, $reps, $ms, $f[8]) |
                Out-File -Append -Encoding utf8 $Out
            # Report each tier against the 8-core CPU, the study's honest baseline.
            if ($tier -eq "T1") { $ref = $ms }
            $vs = ""
            if ($ref -gt 0 -and $tier -match "T2|T3|T4") { $vs = "{0:N2}x vs 8-core CPU" -f ($ref / $ms) }
            Write-Host ("  {0,-9} {1}  {2} ms  {3}" -f $algo, $tier, ("{0:N1}" -f $ms).PadLeft(10), $vs)
        }
    }
    Write-Host ("  elapsed: {0:N1} min" -f ((Get-Date) - $started).TotalMinutes) -ForegroundColor DarkGray
    Write-Host ""
}

Write-Host ("Wrote {0} in {1:N1} min" -f $Out, ((Get-Date) - $started).TotalMinutes) -ForegroundColor Green
