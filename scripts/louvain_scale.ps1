# scripts/louvain_scale.ps1
# How the parallel (T1) Louvain tier behaves as the graph grows: speedup over
# the T0 oracle, and how much community quality that speedup costs.
#
# Louvain is the one algorithm in this study whose parallel tier cannot return
# the oracle's answer (sequential Louvain is order-dependent by construction),
# so both columns matter and neither is reported without the other.
#
# Writes results/louvain_scale.csv.
#
# Usage (from the repo root, after building):
#   powershell -ExecutionPolicy Bypass -File .\scripts\louvain_scale.ps1

param(
    [int[]]$Scales = @(14, 16, 17, 18),
    [int]$EdgeFactor = 16,
    [int]$Threads = 8,
    [int]$Repeats = 3,
    [string]$Bin = ".\build\Release\run_bench.exe",
    [string]$Out = "results\louvain_scale.csv"
)

if (-not (Test-Path $Bin)) { $Bin = ".\build\run_bench.exe" }
if (-not (Test-Path $Bin)) { Write-Error "run_bench not found. Build first (see README)."; exit 1 }

function Get-Field($row, $name) {
    return ((($row[8] -split ($name + '='))[1]) -split ' ')[0]
}

New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null
"scale,vertices,edges,t0_ms,t0_modularity,t1_ms,t1_modularity,speedup,quality_pct" |
    Out-File -Encoding utf8 $Out

foreach ($s in $Scales) {
    $a = (& $Bin --algo louvain --tier seq --rmat $s --ef $EdgeFactor --repeats $Repeats) -split ','
    $env:OMP_NUM_THREADS = $Threads
    $b = (& $Bin --algo louvain --tier openmp --rmat $s --ef $EdgeFactor --repeats $Repeats) -split ','
    Remove-Item Env:\OMP_NUM_THREADS -ErrorAction SilentlyContinue

    $t0 = [double]$a[7]; $q0 = [double](Get-Field $a 'modularity')
    $t1 = [double]$b[7]; $q1 = [double](Get-Field $b 'modularity')
    $sp = $t0 / $t1
    $qp = $q1 / $q0 * 100

    ("{0},{1},{2},{3},{4},{5},{6},{7:N2},{8:N1}" -f `
        $s, $a[3], $a[4], $t0, $q0, $t1, $q1, $sp, $qp) | Out-File -Append -Encoding utf8 $Out
    Write-Host ("2^{0,-3} T0 {1,8:N0} ms Q={2:N4}   T1 {3,8:N0} ms Q={4:N4}   {5:N2}x   quality {6:N1}%" -f `
        $s, $t0, $q0, $t1, $q1, $sp, $qp)
}

Write-Host ("Wrote " + $Out) -ForegroundColor Green
