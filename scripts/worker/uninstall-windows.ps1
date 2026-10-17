# uninstall-windows.ps1 - Stop and remove the scan-worker NSSM service.
# Windows counterpart of uninstall-linux.sh. Relaunches itself elevated (the
# SCM removal needs it), kills any straggling `atomscan worker` processes, and
# leaves the state directory intact, mirroring the Linux uninstall.

param([switch]$Elevated)

$ErrorActionPreference = 'Stop'
$ServiceName = 'scan-worker'
$StateHome   = Join-Path $env:ProgramData 'atomdrift\scan'

function Log($msg) { Write-Host "==> $msg" }

$id = [Security.Principal.WindowsIdentity]::GetCurrent()
$isAdmin = (New-Object Security.Principal.WindowsPrincipal $id).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Log 'Elevating for service removal (UAC prompt)'
    $psExe = (Get-Process -Id $PID).Path
    $p = Start-Process $psExe -Verb RunAs -Wait -PassThru -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $PSCommandPath, '-Elevated')
    if ($p.ExitCode -ne 0) { Write-Host "error: uninstall failed (exit $($p.ExitCode))" -ForegroundColor Red; exit 1 }
    Log 'Uninstall complete'
    Log "Note: state directory $StateHome left intact (remove manually for a fresh state)."
    exit 0
}

if (Get-Service $ServiceName -ErrorAction SilentlyContinue) {
    $nssm = Get-Command nssm -ErrorAction SilentlyContinue
    Log "Stopping and removing $ServiceName"
    if ($nssm) {
        & $nssm.Source stop $ServiceName 2>$null | Out-Null
        & $nssm.Source remove $ServiceName confirm | Out-Null
    } else {
        # nssm gone but its service left behind: plain SCM removal still works.
        Stop-Service $ServiceName -Force -ErrorAction SilentlyContinue
        sc.exe delete $ServiceName | Out-Null
    }
} else {
    Log "No $ServiceName service installed"
}

Log 'Killing any remaining atomscan worker processes'
Get-CimInstance Win32_Process -Filter "Name = 'atomscan.exe'" |
    Where-Object { $_.CommandLine -match 'worker' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

exit 0
