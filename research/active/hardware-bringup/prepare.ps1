param(
    [Parameter(Mandatory)][string]$VendorPackage,
    [Parameter(Mandatory)][string]$ArduinoCli,
    [Parameter(Mandatory)][string]$ArduinoData,
    [string]$Python = 'python',
    [switch]$InstallDependencies
)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$artifactBase = Join-Path $repo 'artifacts.local'
if (-not (Test-Path -LiteralPath $artifactBase -PathType Container)) {
    throw 'Canonical artifacts.local must already exist; do not create a replacement data root.'
}
$out = Join-Path $artifactBase 'hardware-bringup'
New-Item -ItemType Directory -Path $out -Force | Out-Null
$cli = (Resolve-Path -LiteralPath $ArduinoCli).Path
$data = (Resolve-Path -LiteralPath $ArduinoData).Path
$package = (Resolve-Path -LiteralPath $VendorPackage).Path
& $cli version
if ($LASTEXITCODE) { throw 'Arduino CLI is not executable.' }
& $Python --version
if ($LASTEXITCODE) { throw 'Python is not executable.' }
$lock = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'vendor-lock.json') -Raw | ConvertFrom-Json
$vendor = Join-Path $package $lock.source_subdirectory
foreach ($file in $lock.files) {
    $path = Join-Path $vendor $file.path
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing driver: $($file.path)" }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ine $file.sha256) {
        throw "Driver differs from inspected package: $($file.path). Review a new version before updating the lock."
    }
}
$core = Join-Path $data 'packages/m5stack/hardware/esp32/3.3.8'
if (-not (Test-Path -LiteralPath (Join-Path $core 'platform.txt'))) {
    throw 'This pinned build needs m5stack:esp32 3.3.8. Supply its existing Arduino data directory; no hardware operation was performed.'
}
$venv = Join-Path $out 'venv'
$venvPython = Join-Path $venv 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $Python -m venv $venv
    if ($LASTEXITCODE) { throw 'Virtual environment creation failed.' }
}
if ($InstallDependencies) {
    & $venvPython -m pip install --disable-pip-version-check --cache-dir (Join-Path $out 'pip-cache') -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE) { throw 'Dependency installation failed.' }
}
& $venvPython -c 'import serial, esptool; print("Serial and flashing tools import OK")'
if ($LASTEXITCODE) { throw 'Dependencies missing. Rerun with -InstallDependencies.' }
$configuration = [ordered]@{
    arduino_cli = $cli; arduino_data = $data; vendor_package = $package
    python = $venvPython; artifact_root = $out
    fqbn = 'm5stack:esp32:m5stack_atoms3:FlashMode=dio,USBMode=hwcdc,CDCOnBoot=cdc'
    profile_note = 'ESP32-S3 compatibility profile; explicit XIAO pins, no M5 initialization, no PSRAM use.'
}
$configuration | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'local-config.json') -Encoding utf8
& $venvPython -m pip freeze | Set-Content -LiteralPath (Join-Path $out 'environment-freeze.txt') -Encoding utf8
Write-Output "Prepared: $out"
Write-Output 'No serial port opened, device flashed, or network device contacted.'
