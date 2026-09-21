param([ValidateSet('i2c_probe','tof_reader','tof_cnh')][string]$Sketch = 'tof_reader')
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$configPath = Join-Path $repo 'artifacts.local/hardware-bringup/local-config.json'
if (-not (Test-Path -LiteralPath $configPath)) { throw 'Run prepare.ps1 first.' }
$cfg = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$lock = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'vendor-lock.json') -Raw | ConvertFrom-Json
$src = Join-Path $cfg.artifact_root "sketches/$Sketch"
$build = Join-Path $cfg.artifact_root "build/$Sketch"
New-Item -ItemType Directory -Path $src,$build -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "firmware/$Sketch/$Sketch.ino") -Destination $src
$hashes = [ordered]@{}
if ($Sketch -ne 'i2c_probe') {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'firmware/platform.h'),(Join-Path $PSScriptRoot 'firmware/platform.cpp') -Destination $src
    foreach ($file in $lock.files) {
        # Baseline needs no CNH implementation; headers remain identical to the vendor package.
        if ($Sketch -eq 'tof_reader' -and $file.path -eq 'src/vl53lmz_plugin_cnh.c') { continue }
        $path = Join-Path (Join-Path $cfg.vendor_package $lock.source_subdirectory) $file.path
        $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hash -ne $file.sha256) { throw "Driver hash mismatch: $($file.path)" }
        Copy-Item -LiteralPath $path -Destination $src
        $hashes[$file.path] = $hash
    }
}
$cliConfig = Join-Path $cfg.artifact_root 'arduino-cli.json'
@{ directories = @{ data=$cfg.arduino_data; downloads=(Join-Path $cfg.artifact_root 'downloads'); user=(Join-Path $cfg.artifact_root 'arduino-user') } } |
    ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $cliConfig -Encoding utf8
& $cfg.arduino_cli compile --config-file $cliConfig --fqbn $cfg.fqbn --build-path $build $src 2>&1 |
    Tee-Object -FilePath (Join-Path $build 'compile.log')
if ($LASTEXITCODE) { throw "Compilation failed: $Sketch" }
$binary = Join-Path $build "$Sketch.ino.bin"
$sourceHashes = @{}
Get-ChildItem -LiteralPath $src -File | ForEach-Object { $sourceHashes[$_.Name]=(Get-FileHash -LiteralPath $_.FullName).Hash.ToLowerInvariant() }
@{ sketch=$Sketch; fqbn=$cfg.fqbn; source_sha256=$sourceHashes; vendor_sha256=$hashes;
   app_sha256=(Get-FileHash -LiteralPath $binary).Hash.ToLowerInvariant(); app_bytes=(Get-Item -LiteralPath $binary).Length;
   compiled_utc=[DateTime]::UtcNow.ToString('o'); device_operation='none'; binary=$binary } |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Encoding utf8
Write-Output "Built $binary. No upload performed."
