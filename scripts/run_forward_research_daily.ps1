<#
.SYNOPSIS
    Run the forward research daily pipeline unattended and log the result.

.DESCRIPTION
    Intended for Windows Task Scheduler. Credentials are never stored in this
    repository: the script reads ALPHA_VANTAGE_API_KEY, SEC_USER_AGENT and the
    optional MASSIVE_API_KEY from the *user-level* environment (HKCU), so it
    works whether or not the task runs inside an interactive logon session.

    Live capture is requested for consensus and SEC. Market capture is requested
    only when MASSIVE_API_KEY is present. Re-running on the same UTC date reuses
    the immutable completion records and spends no additional quota.

    Logs go to output\daily_logs (ignored by Git). The exit code is the
    pipeline's exit code, so Task Scheduler's "Last Run Result" is meaningful.

.PARAMETER Offline
    Rebuild from preserved data only; makes no request and needs no credentials.
#>
param(
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$logDir = Join-Path $repo "output\daily_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd_HHmmss")
$outLog = Join-Path $logDir "forward-daily-$stamp.log"
$errLog = Join-Path $logDir "forward-daily-$stamp.err.log"

# Load credentials from the user-level registry into this process only.
foreach ($name in "ALPHA_VANTAGE_API_KEY", "SEC_USER_AGENT", "MASSIVE_API_KEY") {
    $value = [Environment]::GetEnvironmentVariable($name, "User")
    if ($value) { Set-Item -Path "Env:$name" -Value $value }
}
# The pipeline prints UTF-8 text; a redirected stdout must not fall back to cp1252.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { $uv = Join-Path $env:USERPROFILE ".local\bin\uv.exe" }
if (-not (Test-Path $uv)) {
    "$(Get-Date -Format o) uv not found; install uv or add it to the user PATH" | Out-File $outLog -Encoding utf8
    exit 3
}

$pipelineArgs = @("run", "python", "-m", "scripts.run_forward_research_daily")
if (-not $Offline) {
    if (-not $env:ALPHA_VANTAGE_API_KEY -or -not $env:SEC_USER_AGENT) {
        "$(Get-Date -Format o) credentials missing: set user-level ALPHA_VANTAGE_API_KEY and SEC_USER_AGENT" |
            Out-File $outLog -Encoding utf8
        exit 2
    }
    $pipelineArgs += "--collect-consensus", "--collect-sec"
    if ($env:MASSIVE_API_KEY) { $pipelineArgs += "--collect-market" }
}

"$(Get-Date -Format o) start offline=$($Offline.IsPresent) args=$($pipelineArgs -join ' ')" |
    Out-File $outLog -Encoding utf8
$process = Start-Process -FilePath $uv -ArgumentList $pipelineArgs -WorkingDirectory $repo `
    -NoNewWindow -Wait -PassThru -RedirectStandardOutput "$outLog.stdout" -RedirectStandardError $errLog
Get-Content "$outLog.stdout" -Encoding utf8 | Out-File $outLog -Append -Encoding utf8
Remove-Item "$outLog.stdout" -Force
"$(Get-Date -Format o) exit=$($process.ExitCode)" | Out-File $outLog -Append -Encoding utf8
if ((Get-Item $errLog).Length -eq 0) { Remove-Item $errLog -Force }

Write-Output "log: $outLog"
exit $process.ExitCode
