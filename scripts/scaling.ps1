# scripts/scaling.ps1
# Thread-scaling benchmark for the OpenMP (T1) tier - run on Windows/PowerShell.
# Measures each algorithm sequential (T0) vs OpenMP at 1/2/4/8/16 threads and
# reports speedup vs the sequential baseline. Writes results/scaling.csv.
#
# Usage (from the repo root, after building):
#   powershell -ExecutionPolicy Bypass -File .\scripts\scaling.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\scaling.ps1 -Scale 20 -EdgeFactor 16

param(
    [int]$Scale = 20,
    [int]$EdgeFactor = 16,
    [int[]]$Threads = @(1,2,4,8,16),
    [string]$Bin = ".\build\Release\run_bench.exe",
    [string]$Out = "results\scaling.csv"
)

# Fall back to the non-MSVC output path if needed.
if (-not (Test-Path $Bin)) { $Bin = ".\build\run_bench.exe" }
if (-not (Test-Path $Bin)) { $Bin = ".\build\run_bench" }
if (-not (Test-Path $Bin)) { Write-Error "run_bench not found. Build first (see README)."; exit 1 }

New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null

$algos = @("bfs","pagerank","wcc")
"algo,tier,threads,median_ms,speedup_vs_seq" | Out-File -Encoding utf8 $Out

foreach ($algo in $algos) {
    Write-Host "== $algo (rmat s$Scale ef$EdgeFactor) ==" -ForegroundColor Cyan

    # Sequential baseline
    $seqRow = & $Bin --algo $algo --tier seq --rmat $Scale --ef $EdgeFactor
    $seqMs  = [double]($seqRow -split ',')[7]
    ("{0},seq,-,{1},1.00" -f $algo, $seqMs) | Out-File -Append -Encoding utf8 $Out
    Write-Host ("  seq        " + ("{0:N1}" -f $seqMs).PadLeft(8) + " ms   (baseline)")

    foreach ($t in $Threads) {
        $env:OMP_NUM_THREADS = $t
        $row = & $Bin --algo $algo --tier openmp --rmat $Scale --ef $EdgeFactor
        $ms  = [double]($row -split ',')[7]
        if ($ms -gt 0) { $sp = $seqMs / $ms } else { $sp = 0 }
        ("{0},openmp,{1},{2},{3:N2}" -f $algo, $t, $ms, $sp) | Out-File -Append -Encoding utf8 $Out
        $line = "  openmp " + ("{0}t" -f $t).PadLeft(3) + "  " + ("{0:N1}" -f $ms).PadLeft(8) + " ms   " + ("{0:N2}x" -f $sp).PadLeft(7)
        Write-Host $line
    }
    Remove-Item Env:\OMP_NUM_THREADS -ErrorAction SilentlyContinue
    Write-Host ""
}

Write-Host ("Wrote " + $Out) -ForegroundColor Green
