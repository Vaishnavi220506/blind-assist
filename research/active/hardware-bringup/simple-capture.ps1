param([int]$Port = 8768)
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$pythonPath = Join-Path $repoRoot 'artifacts.local/hardware-bringup/venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Hardware Python environment missing; run prepare.ps1 first.' }
& $pythonPath -B (Join-Path $PSScriptRoot 'host/simple_capture.py') --port $Port
exit $LASTEXITCODE
