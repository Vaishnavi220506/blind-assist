#requires -Version 7.0
param(
    [string]$Serial,
    [ValidateRange(1024, 65535)][int]$Port = 8766,
    [string]$AdbPath,
    [switch]$CheckOnly,
    [switch]$Disconnect
)
$ErrorActionPreference = 'Stop'
if ($CheckOnly -and $Disconnect) { throw 'Choose CheckOnly or Disconnect.' }
if (-not $AdbPath) {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Android/Sdk/platform-tools/adb.exe'),
        'E:/codex-tools/tools/android-sdk/platform-tools/adb.exe',
        'E:/codex-tools/projects/blindassist/toolchain/android-sdk/platform-tools/adb.exe'
    )
    $found = Get-Command adb.exe -ErrorAction SilentlyContinue
    if ($found) { $AdbPath = $found.Source }
    else { $AdbPath = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1 }
}
if (-not $AdbPath -or -not (Test-Path -LiteralPath $AdbPath)) {
    throw 'ADB executable missing; pass -AdbPath <adb.exe>.'
}
function Invoke-Adb([string[]]$Arguments) {
    $info = [Diagnostics.ProcessStartInfo]::new($AdbPath)
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    foreach ($arg in $Arguments) { $info.ArgumentList.Add($arg) }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $info
    try {
        [void]$process.Start()
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(15000)) {
            $process.Kill()
            throw 'ADB command timed out after 15 seconds.'
        }
        $output = $stdout.GetAwaiter().GetResult().Trim()
        $errorText = $stderr.GetAwaiter().GetResult().Trim()
        if ($process.ExitCode -ne 0) { throw "ADB failed: $errorText $output" }
        return $output
    } finally { $process.Dispose() }
}
$devices = @(foreach ($line in ((Invoke-Adb @('devices')) -split '\r?\n')) {
    if ($line -match '^(\S+)\s+(device|unauthorized|offline)\s*$') {
        [pscustomobject]@{serial=$Matches[1]; state=$Matches[2]}
    }
})
if (-not $Serial) {
    if ($devices.Count -ne 1) { throw 'Connect one Android device or select -Serial explicitly.' }
    $Serial = $devices[0].serial
}
if ($Serial -notmatch '^[A-Za-z0-9._:-]+$') { throw 'Unsupported device serial.' }
$device = $devices | Where-Object serial -EQ $Serial
if (-not $device -or $device.state -ne 'device') {
    throw "Android $Serial is not authorized/ready. Unlock it and allow USB debugging."
}
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$receiptRoot = Join-Path $repoRoot 'artifacts.local/hardware-bringup/phone-view'
$receiptPath = Join-Path $receiptRoot (($Serial -replace ':','_') + "-$Port.json")
$endpoint = "tcp:$Port"
$rows = @((Invoke-Adb @('-s',$Serial,'reverse','--list')) -split '\r?\n' |
    Where-Object { $_.Trim() } | ForEach-Object { ,($_.Trim() -split '\s+') })
$mapping = $rows | Where-Object { $_[1] -eq $endpoint }
if ($mapping -and $mapping[2] -ne $endpoint) {
    throw "Phone port $Port already forwards elsewhere; existing mapping was preserved."
}
$receipt = if (Test-Path -LiteralPath $receiptPath) { Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json }
if ($Disconnect) {
    if (-not $receipt -or -not $receipt.created_by_launcher) {
        throw 'No launcher-owned mapping receipt; no forwarding removed.'
    }
    if ($mapping) { [void](Invoke-Adb @('-s',$Serial,'reverse','--remove',$endpoint)) }
    $receipt.created_by_launcher = $false
    $receipt.disconnected_utc = [DateTime]::UtcNow.ToString('o')
    $receipt | ConvertTo-Json | Set-Content -LiteralPath $receiptPath -Encoding utf8
    [pscustomobject]@{status='disconnected'; serial=$Serial; port=$Port} | ConvertTo-Json
    exit 0
}
$url = "http://127.0.0.1:$Port/"
$state = Invoke-RestMethod "${url}api/state" -TimeoutSec 5
if ($state.mode -notin @('idle','live','replay') -or -not $state.limits) {
    throw 'Port does not identify the hardware dashboard.'
}
if ($CheckOnly) {
    [pscustomobject]@{status='ready'; serial=$Serial; mode=$state.mode; forwarded=[bool]$mapping; url=$url} | ConvertTo-Json
    exit 0
}
$created = -not [bool]$mapping
if ($created) { [void](Invoke-Adb @('-s',$Serial,'reverse','--no-rebind',$endpoint,$endpoint)) }
[void](New-Item -ItemType Directory -Path $receiptRoot -Force)
$owned = $created -or ($receipt -and $receipt.created_by_launcher)
[ordered]@{
    serial=$Serial; port=$Port; url=$url; created_by_launcher=[bool]$owned
    opened_utc=[DateTime]::UtcNow.ToString('o'); disconnected_utc=$null
    path='PC USB collection -> ADB reverse -> phone browser'
} | ConvertTo-Json | Set-Content -LiteralPath $receiptPath -Encoding utf8
try {
    $launch = Invoke-Adb @('-s',$Serial,'shell','am','start','-W','-a','android.intent.action.VIEW','-d',$url)
    if ($launch -match '(?im)^Error:') { throw $launch }
} catch {
    if ($created) {
        [void](Invoke-Adb @('-s',$Serial,'reverse','--remove',$endpoint))
        $failed = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json
        $failed.created_by_launcher = $false
        $failed.disconnected_utc = [DateTime]::UtcNow.ToString('o')
        $failed | ConvertTo-Json | Set-Content -LiteralPath $receiptPath -Encoding utf8
    }
    throw
}
[pscustomobject]@{status='browser_launch_requested'; serial=$Serial; url=$url; receipt=$receiptPath; launch=$launch} | ConvertTo-Json
