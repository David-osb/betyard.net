param(
    [string]$TaskName = "BetYard NBA Nightly Retrain",
    [int]$Hour = 2,
    [int]$Minute = 0,
    [switch]$SkipFetch,
    [switch]$Force,
    [switch]$RunNow
)

$ErrorActionPreference = "Stop"

$pythonExe = "c:\Coding\.venv\Scripts\python.exe"
$scriptPath = "c:\Coding\betyard.net-main\betyard.net-main\ml-backend\nightly_nba_retrain.py"
$startIn = "c:\Coding\betyard.net-main\betyard.net-main\ml-backend"

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found: $pythonExe"
}

if (-not (Test-Path $scriptPath)) {
    throw "Nightly script not found: $scriptPath"
}

if ($Hour -lt 0 -or $Hour -gt 23) {
    throw "Hour must be between 0 and 23"
}

if ($Minute -lt 0 -or $Minute -gt 59) {
    throw "Minute must be between 0 and 59"
}

$inline = "Set-Location '$startIn'; "
if ($SkipFetch) {
    $inline += "`$env:NBA_SKIP_FETCH='1'; "
}
$inline += "& '$pythonExe' '$scriptPath'"

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -Command \"$inline\""
$trigger = New-ScheduledTaskTrigger -Daily -At ([datetime]::Today.AddHours($Hour).AddMinutes($Minute))
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8)

$task = New-ScheduledTask -Action $action -Trigger $trigger -Principal $principal -Settings $settings

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    if (-not $Force) {
        throw "Task '$TaskName' already exists. Re-run with -Force to replace it."
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask -TaskName $TaskName -InputObject $task | Out-Null

Write-Host "Created scheduled task: $TaskName" -ForegroundColor Green
Write-Host "Time: $(('{0:D2}:{1:D2}' -f $Hour, $Minute)) daily" -ForegroundColor Cyan
Write-Host "Python: $pythonExe" -ForegroundColor Gray
Write-Host "Script: $scriptPath" -ForegroundColor Gray
Write-Host "Start in: $startIn" -ForegroundColor Gray
if ($SkipFetch) {
    Write-Host "Mode: Skip fetch enabled (NBA_SKIP_FETCH=1 embedded in scheduled action)" -ForegroundColor Yellow
}

if ($RunNow) {
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "Task started immediately." -ForegroundColor Green
}

Write-Host "Done." -ForegroundColor Green
