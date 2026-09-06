# scripts/louvain_scale.ps1
# How the parallel Louvain tiers behave as the graph grows: T1 (OpenMP) and
# T2 (naive CUDA) against the T0 oracle, in both speed and community quality.
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
"scale,vertices,edges,tier,median_ms,modularity,speedup_vs_t0,quality_pct" |
    Out-File -Encoding utf8 $Out

# tier label -> the --tier flag it maps to
$tiers = [ordered]@{ "T0" = "seq"; "T1" = "openmp"; "T2" = "cuda" }

foreach ($s in $Scales) {
    $t0 = 0.0; $q0 = 0.0
    foreach ($tier in $tiers.Keys) {
        if ($tiers[$tier] -eq "openmp") { $env:OMP_NUM_THREADS = $Threads }
        $f = (& $Bin --algo louvain --tier $tiers[$tier] --rmat $s --ef $EdgeFactor --repeats $Repeats) -split ','
        Remove-Item Env:\OMP_NUM_THREADS -ErrorAction SilentlyContinue
        if ($LASTEXITCODE -ne 0) { Write-Error "run failed: louvain $tier scale $s"; exit 1 }

        $ms = [double]$f[7]
        $q  = [double](Get-Field $f 'modularity')
        if ($tier -eq "T0") { $t0 = $ms; $q0 = $q }
        $sp = $t0 / $ms
        $qp = $q / $q0 * 100

        ("{0},{1},{2},{3},{4},{5},{6:N2},{7:N1}" -f `
            $s, $f[3], $f[4], $tier, $ms, $q, $sp, $qp) | Out-File -Append -Encoding utf8 $Out
        Write-Host ("2^{0,-3} {1}  {2,9:N0} ms   Q={3:N4}   {4,5:N2}x vs one core   quality {5,5:N1}%" -f `
            $s, $tier, $ms, $q, $sp, $qp)
    }
    Write-Host ""
}

Write-Host ("Wrote " + $Out) -ForegroundColor Green
