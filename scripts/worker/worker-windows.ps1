# worker-windows.ps1 - Install Atomdrift Scan worker as an NSSM-supervised
# Windows service.
#
# Windows counterpart of worker-linux.sh. atomscan.exe is a console app and
# cannot talk to the Windows service control manager itself, so NSSM wraps it:
# auto-start at boot, restart-on-failure (10s backoff, mirroring RestartSec=),
# stdout/stderr redirected to rotating log files.
#
# Two phases in one script. The build runs as the invoking user; everything
# that touches ProgramData or the SCM needs elevation, so the script relaunches
# itself once with -ServicePhase via UAC and waits. Settings are forwarded as
# arguments because an elevated process does not inherit the caller's
# environment. Re-runnable: idempotent, and the service is only restarted when
# something actually changed on disk.
#
# Usage: worker-windows.ps1 -Url <url>       (run from the repository root)
#
# Environment overrides (same names as worker-linux.sh):
#   DATA_DIR    local sample dir shared with hopper    (default: unset -> download)
#   WORKERS     concurrency (--workers)                (default: worker auto)
#   MAX_RSS_GB  pause threshold (--max-rss-gb)         (default: unset = worker auto.
#               Unlike Linux there is no MemoryMax= backstop here, so in-process
#               RSS throttling stays ON rather than the systemd-style -1.)
#   LLM / LLM_URL  OpenAI-compatible LLM endpoint or named target (SCAN_LLM)
#   LLM_MODEL      pinned model (SCAN_LLM_MODEL); required for OpenRouter
#   LLM_CONCURRENCY in-flight LLM calls (SCAN_LLM_CONCURRENCY; default 4, vLLM takes 16)
#   SCAN_LLM_KEY   OpenRouter key if ~/.tok/openrouter is absent
#   HOPPER_TOKEN_FILE  hopper API token to install for the service (default: ~/.tok/hopper)
#   LLM_TOKEN_FILE     bearer token for the LLM endpoint            (default: ~/.tok/llm)
#
# The service runs as LocalSystem, not a low-privilege account, on purpose:
# the analysis helpers (rizin, 7z, upx, innoextract) typically live under the
# deploying user's profile (scoop), which LocalService cannot read. The
# deploying user's PATH is baked into the service environment for the same
# reason. Tokens are still copied into the service's own state dir and reached
# via HOME= — never argv, never a secret in the registry env block.

param(
    [string]$Url,
    [switch]$ServicePhase,
    [string]$RepoRoot,
    [string]$DataDir,
    [string]$Workers,
    [string]$MaxRssGb,
    [string]$LlmConcurrency,
    [string]$Llm,
    [string]$LlmModel,
    [string]$HopperTokenFile,
    [string]$LlmTokenFile,
    [string]$OpenRouterKeyFile,   # internal: temp file carrying SCAN_LLM_KEY across the UAC boundary
    [string]$ServicePath          # internal: user PATH captured before elevation
)

$ErrorActionPreference = 'Stop'

$ServiceName = 'scan-worker'
$StateHome   = Join-Path $env:ProgramData 'atomdrift\scan'

function Log($msg) { Write-Host "==> $msg" }
function Die($msg) { Write-Host "error: $msg" -ForegroundColor Red; exit 1 }

function Test-Elevated {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal $id).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Same-content check that tolerates either side being absent.
function Test-SameFile($a, $b) {
    if (-not (Test-Path $a) -or -not (Test-Path $b)) { return $false }
    (Get-FileHash $a).Hash -eq (Get-FileHash $b).Hash
}

# Ports openrouter_target() from worker-linux.sh: does any link of the
# (comma-separated) LLM failover chain point at OpenRouter?
function Test-OpenRouterTarget($chain) {
    foreach ($one in $chain -split ',') {
        $t = $one.Trim()
        if ($t -eq 'openrouter' -or $t -match 'openrouter\.ai') { return $true }
    }
    return $false
}
function Test-OpenRouterOnly($chain) {
    ($chain -notmatch ',') -and (Test-OpenRouterTarget $chain)
}

# ============================================================================
# Phase 2: elevated. Creates state, installs tokens + binary, configures NSSM.
# ============================================================================
if ($ServicePhase) {
    if (-not (Test-Elevated)) { Die 'service phase requires elevation' }
    Set-Location $RepoRoot

    # The elevated console vanishes when this phase exits, so keep a transcript
    # the unelevated parent can print.
    $logDir = Join-Path $StateHome 'logs'
    New-Item -ItemType Directory -Force $logDir | Out-Null
    Start-Transcript -Path (Join-Path $logDir 'deploy-transcript.log') -Force | Out-Null

    try {
        $nssm = (Get-Command nssm -ErrorAction Stop).Source

        # --- State dirs -----------------------------------------------------
        $tokDir = Join-Path $StateHome '.tok'
        foreach ($d in @($StateHome, (Join-Path $StateHome 'bin'),
                         (Join-Path $StateHome 'traits'), $tokDir, $logDir)) {
            New-Item -ItemType Directory -Force $d | Out-Null
        }
        # Lock the token dir down to SYSTEM + Administrators. ProgramData is
        # world-readable by default; the tokens must not be.
        icacls $tokDir /inheritance:r /grant 'SYSTEM:(OI)(CI)F' 'Administrators:(OI)(CI)F' | Out-Null

        # --- Tokens ---------------------------------------------------------
        # Copied into the service's own state (reached via HOME=$StateHome), so
        # a later ACL change on the operator's profile can't strand the
        # service. A rotated token forces a restart below: the worker reads it
        # once, at startup.
        $tokenChanged = $false

        $hopperDst = Join-Path $tokDir 'hopper'
        if ($HopperTokenFile -and (Test-Path $HopperTokenFile) -and (Get-Item $HopperTokenFile).Length -gt 0) {
            if (-not (Test-SameFile $HopperTokenFile $hopperDst)) { $tokenChanged = $true }
            Copy-Item $HopperTokenFile $hopperDst -Force
            Log "Installed hopper API token at $hopperDst"
        } elseif (-not (Test-Path $hopperDst)) {
            Log "WARNING: no hopper API token at $HopperTokenFile; this worker cannot claim work from an authenticated hopper"
        }

        $llmDst = Join-Path $tokDir 'llm'
        if ($LlmTokenFile -and (Test-Path $LlmTokenFile) -and (Get-Item $LlmTokenFile).Length -gt 0) {
            if (-not (Test-SameFile $LlmTokenFile $llmDst)) { $tokenChanged = $true }
            Copy-Item $LlmTokenFile $llmDst -Force
            Log "Installed LLM endpoint token at $llmDst"
        } elseif ($Llm -and -not (Test-OpenRouterTarget $Llm) -and -not (Test-Path $llmDst)) {
            Log "WARNING: no LLM token at $LlmTokenFile; $Llm will refuse the second-opinion pass with 401"
        }

        if (Test-OpenRouterTarget $Llm) {
            if (-not $LlmModel) {
                if (Test-OpenRouterOnly $Llm) { Die 'OpenRouter deploy requires LLM_MODEL= (e.g. qwen/qwen3.8-27b)' }
                Log "WARNING: no LLM_MODEL for the OpenRouter link in $Llm; that link is dropped from the chain"
            }
            $orDst = Join-Path $tokDir 'openrouter'
            $orSrc = Join-Path $env:USERPROFILE '.tok\openrouter'
            if ($OpenRouterKeyFile -and (Test-Path $OpenRouterKeyFile)) {
                # SCAN_LLM_KEY, carried across the UAC boundary in a temp file.
                if (-not (Test-SameFile $OpenRouterKeyFile $orDst)) { $tokenChanged = $true }
                Copy-Item $OpenRouterKeyFile $orDst -Force
                Remove-Item $OpenRouterKeyFile -Force
                Log "Installed OpenRouter token at $orDst"
            } elseif ((Test-Path $orSrc) -and (Get-Item $orSrc).Length -gt 0) {
                if (-not (Test-SameFile $orSrc $orDst)) { $tokenChanged = $true }
                Copy-Item $orSrc $orDst -Force
                Log "Installed OpenRouter token at $orDst"
            } elseif (-not (Test-Path $orDst)) {
                if (Test-OpenRouterOnly $Llm) { Die "OpenRouter deploy needs a key in $orSrc (or SCAN_LLM_KEY)" }
                Log "WARNING: no OpenRouter key in $orSrc (or SCAN_LLM_KEY); that link is dropped from the chain"
            }
        }

        # --- Binary ---------------------------------------------------------
        # Windows locks a running exe against replacement, so a changed binary
        # means stopping the service before the copy. (The service is started
        # or restarted at the end either way.)
        $binSrc = 'target\release\atomscan.exe'
        $binDst = Join-Path $StateHome 'bin\atomscan.exe'
        $svc = Get-Service $ServiceName -ErrorAction SilentlyContinue
        $binaryChanged = -not (Test-SameFile $binSrc $binDst)
        if ($binaryChanged) {
            if ($svc -and $svc.Status -eq 'Running') {
                Log "Stopping $ServiceName to replace the binary"
                & $nssm stop $ServiceName | Out-Null
            }
            Log "Installing $binDst"
            Copy-Item $binSrc "$binDst.new" -Force
            Move-Item "$binDst.new" $binDst -Force
        } else {
            Log 'Binary unchanged'
        }

        # --- Compose worker arguments (mirrors exec_args) --------------------
        $traits = Join-Path $StateHome 'traits'
        $args = "worker --url $Url --traits-dir `"$traits`" --interpret"
        if ($MaxRssGb) { $args += " --max-rss-gb $MaxRssGb" }
        if ($Workers)  { $args += " --workers $Workers" }
        if ($DataDir)  { $args += " --data-dir `"$DataDir`"" }

        # --- NSSM service ----------------------------------------------------
        if (-not $svc) {
            Log "Creating service $ServiceName"
            & $nssm install $ServiceName $binDst | Out-Null
        }
        # `nssm set` is idempotent; run the full configuration every deploy so
        # a changed setting is applied without special-casing.
        & $nssm set $ServiceName Application $binDst | Out-Null
        & $nssm set $ServiceName AppParameters $args | Out-Null
        & $nssm set $ServiceName AppDirectory $StateHome | Out-Null
        & $nssm set $ServiceName DisplayName 'Atomdrift Scan worker' | Out-Null
        & $nssm set $ServiceName Description 'Analyses samples claimed from hopper' | Out-Null
        & $nssm set $ServiceName Start SERVICE_AUTO_START | Out-Null
        # Restart=always / RestartSec=10s. Throttle stops a tight crash loop
        # from pegging the box.
        & $nssm set $ServiceName AppExit Default Restart | Out-Null
        & $nssm set $ServiceName AppRestartDelay 10000 | Out-Null
        & $nssm set $ServiceName AppThrottle 10000 | Out-Null
        # TimeoutStopSec=30: Ctrl-C first, 30s to flush, then harder methods.
        & $nssm set $ServiceName AppStopMethodConsole 30000 | Out-Null
        # journal -> rotating files.
        & $nssm set $ServiceName AppStdout (Join-Path $logDir 'worker.out.log') | Out-Null
        & $nssm set $ServiceName AppStderr (Join-Path $logDir 'worker.err.log') | Out-Null
        & $nssm set $ServiceName AppRotateFiles 1 | Out-Null
        & $nssm set $ServiceName AppRotateOnline 1 | Out-Null
        & $nssm set $ServiceName AppRotateBytes 52428800 | Out-Null

        # HOME steers every ~/.tok lookup (tok_path checks $HOME first) into
        # the state dir — tokens themselves never enter the env block, which
        # any local user can read out of the registry. PATH is the deploying
        # user's, so rizin/7z/upx installed per-user (scoop) stay reachable.
        $svcEnv = @(
            "HOME=$StateHome",
            "USERPROFILE=$StateHome",
            'RUST_BACKTRACE=1',
            "SCAN_LLM=$Llm"
        )
        if ($LlmModel)    { $svcEnv += "SCAN_LLM_MODEL=$LlmModel" }
        # In-flight LLM calls (default 4 in the binary). The endpoint is vLLM,
        # which batches; 16 measured 9 busy cores vs 3 at the default here.
        if ($LlmConcurrency) { $svcEnv += "SCAN_LLM_CONCURRENCY=$LlmConcurrency" }
        if ($ServicePath) { $svcEnv += "PATH=$ServicePath" }
        & $nssm set $ServiceName AppEnvironmentExtra @svcEnv | Out-Null

        # --- Activate --------------------------------------------------------
        $svc = Get-Service $ServiceName
        if ($svc.Status -ne 'Running') {
            Log "Starting $ServiceName"
            & $nssm start $ServiceName | Out-Null
        } elseif ($binaryChanged -or $tokenChanged) {
            Log "Restarting $ServiceName"
            & $nssm restart $ServiceName | Out-Null
        } else {
            Log 'No changes; leaving service running'
        }

        Start-Sleep -Seconds 3
        $svc = Get-Service $ServiceName
        Log "Service status: $($svc.Status)"
        if ($svc.Status -ne 'Running') {
            Die "service failed to start; see $logDir\worker.err.log"
        }
        Log 'Deployment complete'
    } finally {
        Stop-Transcript | Out-Null
    }
    exit 0
}

# ============================================================================
# Phase 1: unelevated. Preconditions + build, then relaunch elevated.
# ============================================================================

if (-not $Url) { Die 'URL required (worker-windows.ps1 -Url <url>)' }
if (-not (Test-Path Makefile)) { Die 'run from the repository root' }
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Die 'cargo not found. Install Rust: winget install Rustlang.Rustup'
}
if (-not (Get-Command rizin -ErrorAction SilentlyContinue)) {
    Die 'rizin not found - install from https://rizin.re first'
}
if (-not (Get-Command nssm -ErrorAction SilentlyContinue)) {
    if (Get-Command scoop -ErrorAction SilentlyContinue) {
        Log 'Installing nssm (scoop)'
        scoop install nssm
        if (-not (Get-Command nssm -ErrorAction SilentlyContinue)) { Die 'scoop install nssm failed' }
    } else {
        Die 'nssm not found - install it (e.g. scoop install nssm) and re-run'
    }
}
foreach ($helper in '7z', 'upx', 'innoextract') {
    if (-not (Get-Command $helper -ErrorAction SilentlyContinue)) {
        Log "warning: $helper not on PATH - samples needing it won't unpack"
    }
}

# Resolve settings from the environment, mirroring worker-linux.sh defaults.
$DataDir  = $env:DATA_DIR
$Workers  = $env:WORKERS
$MaxRssGb = $env:MAX_RSS_GB
$LlmConcurrency = $env:LLM_CONCURRENCY
$Llm      = $env:LLM
if (-not $Llm) { $Llm = $env:LLM_URL }
if (-not $Llm) { $Llm = 'https://llm.isotope13.ai/v1,openrouter' }
$LlmModel = $env:LLM_MODEL
if (-not $LlmModel) { $LlmModel = $env:SCAN_LLM_MODEL }
if (-not $LlmModel) { $LlmModel = ',qwen/qwen3.8-27b' }
$HopperTokenFile = $env:HOPPER_TOKEN_FILE
if (-not $HopperTokenFile) { $HopperTokenFile = Join-Path $env:USERPROFILE '.tok\hopper' }
$LlmTokenFile = $env:LLM_TOKEN_FILE
if (-not $LlmTokenFile) { $LlmTokenFile = Join-Path $env:USERPROFILE '.tok\llm' }

# --- Build (as the invoking user) -------------------------------------------
Log 'Building'
# Scrub the jobserver leak `make` would otherwise hand to build scripts (the
# CARGO= dance from the Makefile).
foreach ($v in 'MAKEFLAGS', 'MAKELEVEL', 'MFLAGS') { Remove-Item "env:$v" -ErrorAction SilentlyContinue }
cargo build --release
if ($LASTEXITCODE -ne 0) { Die 'build failed' }
if (-not (Test-Path 'target\release\atomscan.exe')) { Die 'build did not produce target\release\atomscan.exe' }

# --- Elevate for the service phase ------------------------------------------
# SCAN_LLM_KEY crosses the UAC boundary in a private temp file rather than on
# the (CIM-visible) command line of the elevated process.
$orKeyFile = ''
if ($env:SCAN_LLM_KEY) {
    $orKeyFile = Join-Path $env:TEMP ("scan-orkey-" + [guid]::NewGuid().ToString('N'))
    Set-Content -Path $orKeyFile -Value $env:SCAN_LLM_KEY -NoNewline
}

$fwd = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $PSCommandPath,
         '-ServicePhase',
         '-Url', "`"$Url`"",
         '-RepoRoot', "`"$(Get-Location)`"",
         '-Llm', "`"$Llm`"",
         '-HopperTokenFile', "`"$HopperTokenFile`"",
         '-LlmTokenFile', "`"$LlmTokenFile`"",
         '-ServicePath', "`"$env:PATH`"")
if ($LlmModel)  { $fwd += @('-LlmModel', "`"$LlmModel`"") }
if ($DataDir)   { $fwd += @('-DataDir', "`"$DataDir`"") }
if ($Workers)   { $fwd += @('-Workers', "`"$Workers`"") }
if ($MaxRssGb)  { $fwd += @('-MaxRssGb', "`"$MaxRssGb`"") }
if ($LlmConcurrency) { $fwd += @('-LlmConcurrency', "`"$LlmConcurrency`"") }
if ($orKeyFile) { $fwd += @('-OpenRouterKeyFile', "`"$orKeyFile`"") }

$psExe = (Get-Process -Id $PID).Path
if (Test-Elevated) {
    & $psExe @fwd
    exit $LASTEXITCODE
}

Log 'Elevating for service install (UAC prompt)'
$p = Start-Process $psExe -Verb RunAs -ArgumentList $fwd -Wait -PassThru
# The elevated window is gone; replay its transcript so the outcome is visible.
$transcript = Join-Path $StateHome 'logs\deploy-transcript.log'
if (Test-Path $transcript) {
    Get-Content $transcript | Where-Object { $_ -match '^(==>|error:)' } | ForEach-Object { Write-Host $_ }
}
if ($p.ExitCode -ne 0) { Die "service phase failed (exit $($p.ExitCode)); full transcript: $transcript" }
