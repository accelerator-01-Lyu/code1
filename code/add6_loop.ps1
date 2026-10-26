# add6_loop.ps1 — relaunch add6 until all 72 cells are present in the CSV.
$code = "D:\AL_FuzzyClustering_Paper\v2_unified\code"
$py = "D:\AL_FuzzyClustering_Paper\v2_unified\.venv\Scripts\python.exe"
$csv = Join-Path $code "results\additional\add6_rhobeta_grid.csv"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
Set-Location $code

$maxIters = 40
for ($i = 0; $i -lt $maxIters; $i++) {
    $n = 0
    if (Test-Path $csv) { $n = (Import-Csv $csv).Count }
    Write-Host "=== iter $i : $n cells present ==="
    if ($n -ge 72) { Write-Host "ALL CELLS DONE"; break }
    & $py add6_rhobeta_grid.py *>> add6_loop.log
    Write-Host "=== iter $i exited code $LASTEXITCODE ==="
    Start-Sleep 3
}
Write-Host "LOOP FINISHED"
