# scripts/louvain_chunks.ps1
# Ablation for the one design parameter of the parallel (T1) Louvain tier:
# kChunks, the number of sequential chunks a local-moving pass is split into.
#
# kChunks = 1   is pure Jacobi   -- maximum parallelism, worst communities
# kChunks = n   is Gauss-Seidel  -- the T0 oracle, no parallelism at all
#
# The script rebuilds the library once per value, so it edits
# core/cpu/louvain_omp.cpp in place and restores it afterwards. It exists so the
# table quoted in the README and in louvain_omp.cpp can be regenerated rather
# than trusted.
#
# Usage (from the repo root, after building):
#   powershell -ExecutionPolicy Bypass -File .\scripts\louvain_chunks.ps1

param(
    [int[]]$Chunks = @(1, 4, 16, 64, 256),
    [int]$Scale = 17,
    [int]$EdgeFactor = 16,
    [int]$Threads = 8,
    [int]$Repeats = 3,
    [string]$Src = "core\cpu\louvain_omp.cpp",
    [string]$Bin = ".\build\Release\run_bench.exe",
    [string]$Out = "results\louvain_chunks.csv"
)

if (-not (Test-Path $Bin)) { $Bin = ".\build\run_bench.exe" }
if (-not (Test-Path $Bin)) { Write-Error "run_bench not found. Build first (see README)."; exit 1 }

# Read and write the source with UTF-8 *without* a BOM. Windows PowerShell 5.1's
# `Set-Content -Encoding utf8` emits a BOM, which would silently rewrite the file
# this script is only supposed to borrow.
$srcPath  = (Resolve-Path $Src).Path
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$original = [System.IO.File]::ReadAllText($srcPath)
$pattern  = 'constexpr std::size_t kChunks   = \d+;'

function Set-Chunks([int]$c) {
    $text = [regex]::Replace($original, $pattern, "constexpr std::size_t kChunks   = $c;")
    [System.IO.File]::WriteAllText($srcPath, $text, $utf8NoBom)
}

function Get-Field($row, $name) {
    return ((($row[8] -split ($name + '='))[1]) -split ' ')[0]
}

try {
    New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null
    "tier,chunks,median_ms,modularity,communities" | Out-File -Encoding utf8 $Out

    # T0 reference, measured in the same session on the same graph.
    $r0 = & $Bin --algo louvain --tier seq --rmat $Scale --ef $EdgeFactor --repeats $Repeats
    $f0 = $r0 -split ','
    $q0 = [double](Get-Field $f0 'modularity')
    $t0 = [double]$f0[7]
    ("T0,-,{0},{1},{2}" -f $t0, $q0, (Get-Field $f0 'communities')) | Out-File -Append -Encoding utf8 $Out
    Write-Host ("T0 oracle   {0,9:N0} ms   Q={1:N4}" -f $t0, $q0) -ForegroundColor Cyan

    foreach ($c in $Chunks) {
        Set-Chunks $c
        cmake --build build --config Release -j 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { Write-Error "build failed for kChunks=$c"; exit 1 }

        $env:OMP_NUM_THREADS = $Threads
        $r = & $Bin --algo louvain --tier openmp --rmat $Scale --ef $EdgeFactor --repeats $Repeats
        Remove-Item Env:\OMP_NUM_THREADS -ErrorAction SilentlyContinue
        $f  = $r -split ','
        $ms = [double]$f[7]
        $q  = [double](Get-Field $f 'modularity')
        ("T1,{0},{1},{2},{3}" -f $c, $ms, $q, (Get-Field $f 'communities')) |
            Out-File -Append -Encoding utf8 $Out
        Write-Host ("chunks={0,-5} {1,9:N0} ms   Q={2:N4}   quality={3,5:N1}%   speedup={4:N2}x" -f `
            $c, $ms, $q, ($q / $q0 * 100), ($t0 / $ms))
    }
}
finally {
    # Always put the source back byte-for-byte, even on Ctrl-C or a failure.
    [System.IO.File]::WriteAllText($srcPath, $original, $utf8NoBom)
    cmake --build build --config Release -j 2>&1 | Out-Null
    Write-Host "restored $Src and rebuilt" -ForegroundColor DarkGray
}

Write-Host ("Wrote " + $Out) -ForegroundColor Green
