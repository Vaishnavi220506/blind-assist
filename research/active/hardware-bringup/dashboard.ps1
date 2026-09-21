param(
    [int]$Port = 8766,
    [string]$CameraPort,
    [string]$TofPort
)
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$pythonPath = Join-Path $repoRoot 'artifacts.local/hardware-bringup/venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Hardware Python environment missing; run prepare.ps1 first.'
}
$dashboardArgs = @('-B', (Join-Path $PSScriptRoot 'host/dashboard.py'), '--port', "$Port")
if ($CameraPort) { $dashboardArgs += @('--camera-port', $CameraPort) }
if ($TofPort) { $dashboardArgs += @('--tof-port', $TofPort) }
& $pythonPath @dashboardArgs
exit $LASTEXITCODE
