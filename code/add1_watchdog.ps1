# add1_watchdog.ps1 — relaunch add1_run_budgets.py whenever it dies, until the
# log reports ALL DONE. The runner is resumable (skips completed job keys).
$py = "D:\AL_FuzzyClustering_Paper\v2_unified\.venv\Scripts\python.exe"
$wd = "D:\AL_FuzzyClustering_Paper\v2_unified\code"
$log = "D:\AL_FuzzyClustering_Paper\v2_unified\code\results\additional\_add1_run.log"
$n = 0
while ($true) {
    $done = Select-String -Path $log -Pattern "ALL DONE" -Quiet
    if ($done) { break }
    $n++
    Write-Host ("[watchdog] launch #{0} at {1}" -f $n, (Get-Date -Format "HH:mm:ss"))
    $p = Start-Process -FilePath $py -ArgumentList "add1_run_budgets.py" `
        -WorkingDirectory $wd -Wait -PassThru `
        -RedirectStandardOutput "$wd\results\additional\_add1_stdout.log" `
        -RedirectStandardError  "$wd\results\additional\_add1_stderr.log"
    Write-Host ("[watchdog] exit code {0} at {1}" -f $p.ExitCode, (Get-Date -Format "HH:mm:ss"))
    Start-Sleep -Seconds 3
}
Write-Host "[watchdog] ALL DONE detected; watchdog exiting."
