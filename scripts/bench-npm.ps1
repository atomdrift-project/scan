# Usage: pwsh -File out/bench-npm.ps1 <id> [exePath] [dataset] [extraEnv "A=1,B=2"]
param([Parameter(Mandatory=$true)][string]$Id,
      [string]$Exe = 'C:\src\scan\out\atomscan.exe',
      [string]$Data = 'C:\data\npm-mspd',
      [string]$EnvPairs = '',
      [string]$Traits = '',
      [string]$Mode = 'balanced')
$env:CLEAVE_SKIP_CACHE = '1'
$env:CLEAVE_SKIP_YARA_CACHE = '0'
$env:SCAN_FETCH = 'none'
$env:SCAN_NO_UPDATE_CHECK = '1'
# Rule set: explicit -Traits wins; otherwise an inherited CLEAVE_TRAITS_DIR is
# honoured, and only a bare invocation falls back to the installed bundle.
if ($Traits) { $env:CLEAVE_TRAITS_DIR = $Traits }
elseif (-not $env:CLEAVE_TRAITS_DIR) { Remove-Item Env:CLEAVE_TRAITS_DIR -ErrorAction SilentlyContinue }
"Rules: " + $(if ($env:CLEAVE_TRAITS_DIR) { $env:CLEAVE_TRAITS_DIR } else { "installed bundle" }) | Write-Host
foreach ($pair in $EnvPairs -split ',') {
  if ($pair -match '^([^=]+)=(.*)$') { Set-Item -Path ("Env:" + $Matches[1]) -Value $Matches[2] }
}
$out = "C:\src\scan\out\$Id.json"
$err = "C:\src\scan\out\$Id.err"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$p = Start-Process -FilePath $Exe -ArgumentList @('-f','json','--mode',$Mode,'--no-update','--fetch=none',$Data) -RedirectStandardOutput $out -RedirectStandardError $err -PassThru -NoNewWindow
# Sample the live working set at 100 ms and keep the maximum. `PeakWorkingSet64`
# is not usable here: it reads 0 after exit, and while running it silently
# stops updating once the handle can no longer be refreshed, which under-read
# a 6 GB run as 2 GB. Failures are counted, not swallowed.
$peakBytes = 0
$samples = 0
$sampleErrors = 0
while (-not $p.HasExited) {
  # CIM `Win32_Process.WorkingSetSize`. Every other source lied here:
  # `PeakWorkingSet64` reads 0 after exit and stops updating while the child
  # is busy (under-read a 4 GB run as 2 GB), and the engine's own
  # `peak_rss_mb` reports less than its own `current_rss_mb`.
  try {
    $ws = (Get-CimInstance Win32_Process -Filter "ProcessId=$($p.Id)" -ErrorAction Stop).WorkingSetSize
    $peakBytes = [math]::Max($peakBytes, [int64]$ws); $samples++
  } catch { $sampleErrors++ }
  Start-Sleep -Milliseconds 100
}
$p.WaitForExit()
$sw.Stop()
$cpu = $p.TotalProcessorTime.TotalSeconds
# `Process.PeakWorkingSet64` reads 0 once the process has exited (the handle's
# counters are gone), so sample it from the job object's peak instead: poll
# while it runs and keep the maximum. Cheap at 200 ms.
$peak = [math]::Round($peakBytes/1MB)
# The engine also tracks its own RSS; when it logged a peak, report that too.
$selfPeak = (Select-String -Path $err -Pattern 'peak_rss_mb=(\d+)' -AllMatches -ErrorAction SilentlyContinue |
  ForEach-Object { $_.Matches } | ForEach-Object { [int]$_.Groups[1].Value } | Measure-Object -Maximum).Maximum
"{0}: wall={1:N2}s cpu={2:N1}s peakWS={3}MB selfPeak={6}MB exit={4} json={5} samples={7}/{8}err" -f $Id, $sw.Elapsed.TotalSeconds, $cpu, $peak, $p.ExitCode, (Get-Item $out).Length, $selfPeak, $samples, $sampleErrors
