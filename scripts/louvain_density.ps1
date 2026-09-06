# scripts/louvain_density.ps1
# The measurement that decided the T4 design: how the three GPU groupings
# behave as the graph gets denser.
#
#   T2  global 64-bit radix sort of (vertex, community) over every arc
#   T3  the same sort, but only over the bits a composite key can occupy
#   T4  no sort at all -- an open-addressed hash table per vertex
#
# T4's first version sized its table by degree *bucket*, and this sweep is what
# caught the consequence: clearing and scanning cost capacity, not degree, so a
# degree-64 vertex landing in the 2048-slot bucket paid 32x for nothing, and T4
# collapsed to 0.44x of T3 at average degree 256. Sizing the table from the
# vertex's own degree fixed it. Re-run this before changing any bucket bound.
#
# Writes results/louvain_density.csv.
#
# Usage (from the repo root, after building):
#   powershell -ExecutionPolicy Bypass -File .\scripts\louvain_density.ps1

param(
    [int]$Scale = 16,
    [int[]]$EdgeFactors = @(8, 16, 32, 64, 128),
    [int]$Repeats = 3,
    [string]$Bin = ".\build\Release\run_bench.exe",
    [string]$Out = "results\louvain_density.csv"
)

if (-not (Test-Path $Bin)) { $Bin = ".\build\run_bench.exe" }
if (-not (Test-Path $Bin)) { Write-Error "run_bench not found. Build first (see README)."; exit 1 }

function Get-Field($row, $name) {
    return ((($row[8] -split ($name + '='))[1]) -split ' ')[0]
}

New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null
"scale,edge_factor,avg_degree,tier,median_ms,modularity" | Out-File -Encoding utf8 $Out

$tiers = [ordered]@{ "T2" = "cuda"; "T3" = "cuda_opt"; "T4" = "cuda_lb" }

Write-Host ("{0,9}  {1,9}  {2,9}  {3,9}   {4}" -f "avg_deg", "T2", "T3", "T4", "T4 vs T3") -ForegroundColor Cyan
foreach ($ef in $EdgeFactors) {
    $ms = @{}
    $avg = 0
    foreach ($tier in $tiers.Keys) {
        $f = (& $Bin --algo louvain --tier $tiers[$tier] --rmat $Scale --ef $ef --repeats $Repeats) -split ','
        if ($LASTEXITCODE -ne 0) { Write-Error "run failed: louvain $tier ef $ef"; exit 1 }
        $ms[$tier] = [double]$f[7]
        $avg = [math]::Round([double]$f[6], 0)
        ("{0},{1},{2},{3},{4},{5}" -f $Scale, $ef, $avg, $tier, $f[7], (Get-Field $f 'modularity')) |
            Out-File -Append -Encoding utf8 $Out
    }
    Write-Host ("{0,9}  {1,9:N0}  {2,9:N0}  {3,9:N0}   {4:N2}x" -f `
        $avg, $ms["T2"], $ms["T3"], $ms["T4"], ($ms["T3"] / $ms["T4"]))
}

Write-Host ("Wrote " + $Out) -ForegroundColor Green
