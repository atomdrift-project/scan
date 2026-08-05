<#
.SYNOPSIS
    Install Atomdrift Scan (the `atomscan` CLI) on Windows.

.DESCRIPTION
    irm https://install.atomdrift.org/ps1 | iex

    With options, which `iex` cannot pass through:

    & ([scriptblock]::Create((irm https://install.atomdrift.org/ps1))) -Method Binary

    It works out the platform, fetches a release binary (checksum- and
    provenance-verified), falls back to a source build when no binary is
    published for it, puts the install directory on PATH, and reports on the
    optional analysis tools that make scans deeper.

    Re-running is safe and cheap: an install that is already current is left
    alone, and the binary is replaced atomically.

    Windows PowerShell 5.1 (which ships with Windows 10 and 11) and
    PowerShell 7 are both supported.

.PARAMETER Version
    Install a specific version. Defaults to the latest release.

.PARAMETER Dir
    Install into this directory. Defaults to %LOCALAPPDATA%\Programs\atomscan\bin.

.PARAMETER Method
    Auto, Binary, or Source. Auto tries a release binary, then a source build.

.PARAMETER NoTools
    Skip the optional analysis tool check (rizin, upx, 7-Zip, innoextract).

.PARAMETER NoPath
    Do not add the install directory to the user PATH.

.PARAMETER Force
    Reinstall even when the target version is already there.

.PARAMETER Quiet
    Only report problems.

.LINK
    https://github.com/atomdrift-project/scan
#>

[CmdletBinding()]
# Write-Host is the point here: this is a terminal UI, and its output must not
# land in a caller's pipeline. The rest are analyzer heuristics that misread an
# installer: the parameters are read inside script-scoped functions, and no
# function here changes system state without the caller having asked it to.
[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '')]
[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSReviewUnusedParameter', '')]
[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '')]
[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseSingularNouns', '')]
param(
	[string]$Version = $env:ATOMSCAN_VERSION,
	[string]$Dir = $env:ATOMSCAN_INSTALL_DIR,
	[ValidateSet('Auto', 'Binary', 'Source')]
	[string]$Method = 'Auto',
	[switch]$NoTools,
	[switch]$NoPath,
	[switch]$Force,
	[switch]$Quiet
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
# Invoke-WebRequest's own progress bar is both slow and redundant here.
$ProgressPreference = 'SilentlyContinue'

$Repo = 'atomdrift-project/scan'
$BinName = 'atomscan'
$ExeName = 'atomscan.exe'

# Targets published by .github/workflows/release.yml. Windows is not among them
# yet: its build job runs but is not expected to produce an artifact, so a
# Windows install is a source build until that changes.
$Targets = @(
	'x86_64-unknown-linux-gnu', 'aarch64-unknown-linux-gnu'
	'x86_64-unknown-linux-musl', 'aarch64-unknown-linux-musl'
	's390x-unknown-linux-gnu', 'riscv64gc-unknown-linux-gnu'
	'powerpc64le-unknown-linux-gnu'
	'aarch64-apple-darwin', 'x86_64-apple-darwin'
	'x86_64-unknown-freebsd', 'x86_64-unknown-openbsd'
	'x86_64-unknown-netbsd', 'x86_64-unknown-dragonfly'
	'x86_64-unknown-illumos'
)

# Filled in as we go.
$script:Self = ''         # this machine's target triple, published or not
$script:Target = ''       # same, but empty unless a release carries it
$script:Resolved = ''     # version being installed
$script:InstallDir = ''
$script:Installed = ''    # full path of the binary we installed
$script:Changed = $false  # whether this run replaced anything
$script:Temp = ''

# ---------------------------------------------------------------------------
# Style
#
# ANSI when the host can render it, plain text otherwise; UTF-8 art when the
# console will take it, ASCII otherwise. The palette is the six-bar spectrum
# from media/logo.svg.
# ---------------------------------------------------------------------------

function Initialize-Style {
	$ansi = $false
	if (-not $env:NO_COLOR) {
		if ($env:WT_SESSION) {
			$ansi = $true
		} else {
			try { $ansi = [bool]$Host.UI.SupportsVirtualTerminal } catch { $ansi = $false }
		}
	}

	$e = [char]27
	if ($ansi) {
		$script:CRed = "$e[38;2;226;75;74m"
		$script:COrange = "$e[38;2;216;90;48m"
		$script:CAmber = "$e[38;2;239;159;39m"
		$script:CGreen = "$e[38;2;99;153;34m"
		$script:CTeal = "$e[38;2;29;158;117m"
		$script:CBlue = "$e[38;2;55;138;221m"
		$script:CDim = "$e[2m"
		$script:CBold = "$e[1m"
		$script:CReset = "$e[0m"
	} else {
		$script:CRed = '' ; $script:COrange = '' ; $script:CAmber = ''
		$script:CGreen = '' ; $script:CTeal = '' ; $script:CBlue = ''
		$script:CDim = '' ; $script:CBold = '' ; $script:CReset = ''
	}

	# Box-drawing and block elements only: unlike the geometric shapes they are
	# unambiguously one cell wide, so the art cannot smear. They are written as
	# [char] escapes rather than literals to keep this file pure ASCII, which is
	# what stops Windows PowerShell 5.1 mangling it when it is saved without a
	# byte-order mark.
	$script:Utf8 = $false
	try {
		[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
		$script:Utf8 = $true
	} catch {
		$script:Utf8 = $false
	}
	if ($script:Utf8) {
		$script:GStep = [string][char]0x25B8 ; $script:GOk = [string][char]0x2713
		$script:GWarn = '!' ; $script:GErr = [string][char]0x2717
		$script:GBar = [string][char]0x2501 * 2
		$script:GFull = [string][char]0x2588 ; $script:GEmpty = [string][char]0x2591
		# Orbit: top-left, top-right, bottom-left, bottom-right, arc, side, nucleus.
		$script:ATl = [string][char]0x256D ; $script:ATr = [string][char]0x256E
		$script:ABl = [string][char]0x2570 ; $script:ABr = [string][char]0x256F
		$script:AH = [string][char]0x2500 ; $script:AV = [string][char]0x2502
		$script:ANuc = [string][char]0x2588 * 2
		$script:GDot = [string][char]0x00B7
	} else {
		$script:GStep = '>' ; $script:GOk = '+'
		$script:GWarn = '!' ; $script:GErr = 'x'
		$script:GBar = '==' ; $script:GFull = '#' ; $script:GEmpty = '.'
		$script:ATl = '.' ; $script:ATr = '.'
		$script:ABl = "'" ; $script:ABr = "'"
		$script:AH = '-' ; $script:AV = '|'
		$script:ANuc = '**'
		$script:GDot = '-'
	}

	$script:StepN = 0
	# The progress bar redraws with a carriage return, which is meaningless in a
	# redirected log. This is the PowerShell spelling of `[ -t 1 ]`.
	$redirected = $false
	try { $redirected = [Console]::IsOutputRedirected } catch { $redirected = $false }
	$script:Interactive = (-not $redirected) -and (-not $Quiet)
}

function Get-StepColor {
	$script:StepN++
	switch ($script:StepN % 6) {
		1 { $script:CRed }
		2 { $script:COrange }
		3 { $script:CAmber }
		4 { $script:CGreen }
		5 { $script:CTeal }
		default { $script:CBlue }
	}
}

function Write-Step([string]$Label, [string]$Value) {
	if ($Quiet) { return }
	$c = Get-StepColor
	Write-Host (" {0}{1}{2} {3}{4}{5}{6}" -f $c, $script:GStep, $script:CReset,
		$script:CDim, $Label.PadRight(11), $script:CReset, $Value)
}

function Write-Ok([string]$Label, [string]$Value) {
	if ($Quiet) { return }
	Write-Host (" {0}{1}{2} {3}{4}{5}{6}" -f $script:CGreen, $script:GOk, $script:CReset,
		$script:CDim, $Label.PadRight(11), $script:CReset, $Value)
}

function Write-Note([string]$Text) {
	if ($Quiet) { return }
	Write-Host ("   {0}{1}{2}" -f $script:CDim, $Text, $script:CReset)
}

function Write-Warn([string]$Text) {
	Write-Host (" {0}{1}{2} {3}" -f $script:CAmber, $script:GWarn, $script:CReset, $Text)
}

function Stop-Install([string]$Text) {
	Write-Host ''
	Write-Host (" {0}{1}{2} {3}{4}{5}" -f $script:CRed, $script:GErr, $script:CReset,
		$script:CBold, $Text, $script:CReset)
	Write-Host ''
	exit 1
}

# The logo's six-bar spectrum, in text.
function Get-SpectrumRule {
	($script:CRed + $script:GBar + ' ' + $script:COrange + $script:GBar + ' ' +
		$script:CAmber + $script:GBar + ' ' + $script:CGreen + $script:GBar + ' ' +
		$script:CTeal + $script:GBar + ' ' + $script:CBlue + $script:GBar + $script:CReset)
}

# An atom: a nucleus inside an orbit drawn in the six brand colours, running
# clockwise from red at ten o'clock round to blue at nine.
function Write-Banner {
	if ($Quiet) { return }
	$tag = 'static analysis + local ML malware detection'
	$arc = $script:AH * 4
	$gap = ' ' * 8
	Write-Host ''
	Write-Host ("   {0}{1}{2}{3}{4}{5}{6}" -f
		$script:CRed, $script:ATl, $arc, $script:COrange, $arc, $script:ATr, $script:CReset)
	Write-Host (" {0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}    {11}atomdrift scan{12}" -f
		$script:CRed, $script:ATl, $script:AH, $script:ABr, $script:CReset, $gap,
		$script:CAmber, $script:ABl, $script:AH, $script:ATr, $script:CReset,
		$script:CBold, $script:CReset)
	Write-Host (" {0}{1}{2}     {3}{4}{5}{6}     {7}{8}{9}    {10}" -f
		$script:CBlue, $script:AV, $script:CReset, $script:CBold, $script:CAmber, $script:ANuc,
		$script:CReset, $script:CAmber, $script:AV, $script:CReset, (Get-SpectrumRule))
	Write-Host (" {0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}    {11}{12}{13}" -f
		$script:CTeal, $script:ABl, $script:AH, $script:ATr, $script:CReset, $gap,
		$script:CGreen, $script:ATl, $script:AH, $script:ABr, $script:CReset,
		$script:CDim, $tag, $script:CReset)
	Write-Host ("   {0}{1}{2}{3}{4}{5}{6}" -f
		$script:CTeal, $script:ABl, $arc, $script:CGreen, $arc, $script:ABr, $script:CReset)
	Write-Host ''
}

# Every target a release carries, two to a row under the `targets` label, with
# this machine's own picked out. Windows has no published binary yet, so it is
# listed at the end as the source build it is about to get.
function Write-TargetList {
	if ($Quiet) { return }
	$label = 'targets'
	$line = ''
	$mine = $false
	$n = 0
	foreach ($t in $Targets) {
		# Pad the left column only, so rows carry no trailing whitespace. The
		# padding goes on the bare text, before any colour: escape sequences are
		# characters as far as PadRight is concerned.
		if ($t -eq $script:Self) {
			$mine = $true
			$text = $script:GStep + ' ' + $t
			$colour = $script:CBold + $script:CTeal
		} else {
			$text = '  ' + $t
			$colour = $script:CDim
		}
		if ($n % 2 -eq 0) { $text = $text.PadRight(32) }
		$line += $colour + $text + $script:CReset
		$n++
		if ($n % 2 -eq 0) {
			Write-TargetRow $label $line $mine
			$label = '' ; $line = '' ; $mine = $false
		}
	}
	if ($line) { Write-TargetRow $label $line $mine }

	if (-not $script:Target) {
		$cell = $script:CBold + $script:CAmber + $script:GStep + ' ' + $script:Self +
		$script:CReset + $script:CAmber + '  (source-based install)' + $script:CReset
		Write-TargetRow '' $cell $false
	}
}

function Write-TargetRow([string]$Label, [string]$Cells, [bool]$Mine) {
	$row = "   " + $script:CDim + $Label.PadRight(11) + $script:CReset + $Cells
	if ($Mine) { $row += $script:CDim + '  this machine' + $script:CReset }
	Write-Host $row
}

# ---------------------------------------------------------------------------
# Platform
# ---------------------------------------------------------------------------

function Resolve-Platform {
	$arch = $env:PROCESSOR_ARCHITECTURE
	if ($env:PROCESSOR_ARCHITEW6432) { $arch = $env:PROCESSOR_ARCHITEW6432 }
	switch ($arch) {
		'AMD64' { $script:Self = 'x86_64-pc-windows-msvc' }
		'ARM64' { $script:Self = 'aarch64-pc-windows-msvc' }
		'x86' { $script:Self = 'i686-pc-windows-msvc' }
		default { $script:Self = "$arch-pc-windows-msvc" }
	}
	$script:Target = ''
	if ($Targets -contains $script:Self) { $script:Target = $script:Self }

	$os = 'Windows'
	try {
		$os = (Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).Caption.Trim()
	} catch {
		try { $os = [System.Environment]::OSVersion.VersionString } catch { $os = 'Windows' }
	}
	Write-Step 'platform' ("{0}  {1}{2}{3}" -f $script:Self, $script:CDim, $os, $script:CReset)
}

# ---------------------------------------------------------------------------
# HTTP
#
# HttpWebRequest rather than Invoke-WebRequest: it is present in Windows
# PowerShell 5.1 and PowerShell 7 alike, it streams, and it reports the length
# up front, which is what the progress bar is drawn from.
# ---------------------------------------------------------------------------

function Initialize-Tls {
	try {
		[Net.ServicePointManager]::SecurityProtocol =
		[Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13
	} catch {
		# .NET Framework 4.7 and earlier have no Tls13 member to name.
		try {
			[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
		} catch {
			Write-Verbose "leaving SecurityProtocol at its default: $_"
		}
	}
}

function New-Request([string]$Url) {
	$req = [System.Net.HttpWebRequest]::Create($Url)
	$req.UserAgent = "atomscan-install/1.0 (PowerShell $($PSVersionTable.PSVersion))"
	$req.Timeout = 30000
	$req.ReadWriteTimeout = 120000
	return $req
}

# The newest release tag, read from the redirect /releases/latest performs. The
# REST API would do too, but it is rate-limited per IP, which is exactly the
# wrong failure mode on a shared CI address.
function Resolve-LatestVersion {
	try {
		$req = New-Request "https://github.com/$Repo/releases/latest"
		$req.Method = 'HEAD'
		$resp = $req.GetResponse()
		$uri = $resp.ResponseUri.AbsoluteUri
		$resp.Dispose()
		if ($uri -match '/releases/tag/(.+)$') { return $Matches[1].TrimStart('v') }
	} catch {
		Write-Verbose "latest-release redirect failed: $_"
	}
	return ''
}

# Deliberately narrow: only a 404 counts as absent. A HEAD refused for any other
# reason must not be read as "this platform has no binary", or a working release
# would quietly become a very long source build.
function Test-UrlExists([string]$Url) {
	try {
		$req = New-Request $Url
		$req.Method = 'HEAD'
		$resp = $req.GetResponse()
		$resp.Dispose()
		return $true
	} catch [System.Net.WebException] {
		$response = $_.Exception.Response
		if ($response -and [int]$response.StatusCode -eq 404) { return $false }
		return $true
	} catch {
		return $true
	}
}

function Format-Size([long]$Bytes) {
	if ($Bytes -ge 1048576) { return ('{0:N1} MB' -f ($Bytes / 1048576)) }
	if ($Bytes -ge 1024) { return ('{0:N0} KB' -f ($Bytes / 1024)) }
	return "$Bytes B"
}

function Write-Bar([long]$Done, [long]$Total) {
	$width = 18
	$bar = ''
	if ($Total -gt 0) {
		$pct = [math]::Min(100, [int](($Done * 100) / $Total))
		$fill = [int](($pct * $width) / 100)
		$bar = ($script:GFull.ToString() * $fill) + ($script:GEmpty.ToString() * ($width - $fill))
		$text = " {0}{1}{2} {3}{4}{5}{6}{7}{8} {9,3}%  {10}" -f
		$script:CTeal, $script:GStep, $script:CReset, $script:CDim, 'download'.PadRight(11),
		$script:CReset, $script:CTeal, $bar, $script:CReset, $pct, (Format-Size $Done)
	} else {
		$text = " {0}{1}{2} {3}{4}{5}{6}" -f
		$script:CTeal, $script:GStep, $script:CReset, $script:CDim, 'download'.PadRight(11),
		$script:CReset, (Format-Size $Done)
	}
	Write-Host "`r$text" -NoNewline
}

function Save-Url([string]$Url, [string]$Path, [string]$Label) {
	$resp = $null ; $in = $null ; $out = $null
	try {
		$resp = (New-Request $Url).GetResponse()
		$total = $resp.ContentLength
		$in = $resp.GetResponseStream()
		$out = [System.IO.File]::Create($Path)
		$buffer = New-Object byte[] 131072
		$done = [long]0
		$lastDraw = [long]0
		while ($true) {
			$read = $in.Read($buffer, 0, $buffer.Length)
			if ($read -le 0) { break }
			$out.Write($buffer, 0, $read)
			$done += $read
			# Redrawing on every 64 KB chunk would spend more time in the
			# console than on the network.
			if ($script:Interactive -and ($done - $lastDraw) -gt 262144) {
				Write-Bar $done $total
				$lastDraw = $done
			}
		}
	} finally {
		if ($out) { $out.Dispose() }
		if ($in) { $in.Dispose() }
		if ($resp) { $resp.Dispose() }
	}
	if ($script:Interactive) { Write-Host ("`r" + (' ' * 78) + "`r") -NoNewline }
	$size = (Get-Item $Path).Length
	Write-Ok 'download' ("{0}  {1}{2}{3}" -f $Label, $script:CDim, (Format-Size $size), $script:CReset)
}

# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------

# Fails closed. The digest travels over the same TLS connection as the archive,
# so this is a corruption and truncation check; the attestation below is the
# trust anchor.
function Test-Checksum([string]$File, [string]$SumsFile, [string]$Name) {
	$want = ''
	foreach ($line in [System.IO.File]::ReadAllLines($SumsFile)) {
		$fields = $line -split '\s+', 2
		if ($fields.Count -lt 2) { continue }
		$listed = $fields[1].Trim()
		if ($listed.StartsWith('./')) { $listed = $listed.Substring(2) }
		if ($listed.StartsWith('*')) { $listed = $listed.Substring(1) }
		if ($listed -eq $Name) { $want = $fields[0].Trim().ToLowerInvariant(); break }
	}
	if (-not $want) { Stop-Install "$Name is not listed in SHA256SUMS - refusing to install" }

	$got = (Get-FileHash -Algorithm SHA256 -LiteralPath $File).Hash.ToLowerInvariant()
	if ($got -ne $want) {
		Stop-Install "checksum mismatch for $Name - refusing to install`n   expected $want`n   got      $got"
	}
	return $got.Substring(0, 12)
}

# Signed build provenance, when the GitHub CLI is here to check it. Its absence
# is not an error: this is a stronger check than we can otherwise make, not a
# required one.
function Test-Provenance([string]$File) {
	if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { return $false }
	try {
		& gh attestation verify $File --repo $Repo *> $null
		return $LASTEXITCODE -eq 0
	} catch {
		return $false
	}
}

# ---------------------------------------------------------------------------
# Install location
#
# %LOCALAPPDATA%\Programs is where a per-user install belongs on Windows: no
# administrator rights, no UAC prompt, and nothing left in Program Files for
# another user to trip over.
# ---------------------------------------------------------------------------

function Resolve-InstallDir {
	if ($Dir) {
		$script:InstallDir = $Dir
	} else {
		$existing = Get-Command $BinName -ErrorAction SilentlyContinue
		if ($existing -and $existing.Source) {
			# Upgrading in place beats installing a second copy that shadows the
			# first.
			$script:InstallDir = Split-Path -Parent $existing.Source
		} else {
			$script:InstallDir = Join-Path $env:LOCALAPPDATA "Programs\$BinName\bin"
		}
	}
	if (-not (Test-Path -LiteralPath $script:InstallDir)) {
		New-Item -ItemType Directory -Force -Path $script:InstallDir | Out-Null
	}
}

function Get-InstalledVersion([string]$Path) {
	if (-not (Test-Path -LiteralPath $Path)) { return '' }
	try {
		$out = & $Path --version 2>$null
		if ($LASTEXITCODE -ne 0) { return '' }
		return ("$out".Trim() -split '\s+')[1]
	} catch {
		return ''
	}
}

# Replace the binary by rename rather than by overwrite: a reader sees either
# the old file or the new one, and a running atomscan.exe cannot be written to
# in place on Windows at all.
function Install-Binary([string]$Source) {
	$dest = Join-Path $script:InstallDir $ExeName
	$new = "$dest.new"
	$old = "$dest.old"
	Copy-Item -LiteralPath $Source -Destination $new -Force
	if (Test-Path -LiteralPath $dest) {
		Remove-Item -LiteralPath $old -Force -ErrorAction SilentlyContinue
		Move-Item -LiteralPath $dest -Destination $old -Force
	}
	try {
		Move-Item -LiteralPath $new -Destination $dest -Force
	} catch {
		if (Test-Path -LiteralPath $old) { Move-Item -LiteralPath $old -Destination $dest -Force }
		Stop-Install "cannot replace $dest - is atomscan running?"
	}
	# A previous copy can stay locked by a running process; it is not fatal.
	Remove-Item -LiteralPath $old -Force -ErrorAction SilentlyContinue
	$script:Installed = $dest
	$script:Changed = $true
}

# ---------------------------------------------------------------------------
# Method: release binary
#
# Returns $false when this platform has no published binary - the signal for
# main to fall back to a source build.
# ---------------------------------------------------------------------------

function Install-FromRelease {
	if (-not $script:Target) {
		Write-Warn 'no published binary for Windows yet'
		return $false
	}

	if ($Version) {
		$script:Resolved = $Version.TrimStart('v')
	} else {
		$script:Resolved = Resolve-LatestVersion
		if (-not $script:Resolved) {
			Write-Warn 'could not work out the latest release'
			return $false
		}
	}
	$pinned = ''
	if ($Version) { $pinned = "  $($script:CDim)(pinned)$($script:CReset)" }
	Write-Step 'version' "$($script:Resolved)$pinned"

	# Idempotence: an install that is already what we would install is done.
	$dest = Join-Path $script:InstallDir $ExeName
	if (-not $Force -and (Get-InstalledVersion $dest) -eq $script:Resolved) {
		$script:Installed = $dest
		Write-Ok 'up to date' "$dest  $($script:CDim)$($script:Resolved)$($script:CReset)"
		return $true
	}

	$name = "$BinName-$($script:Resolved)-$($script:Target).tar.gz"
	$base = "https://github.com/$Repo/releases/download/v$($script:Resolved)"
	if (-not (Test-UrlExists "$base/$name")) {
		Write-Warn "release v$($script:Resolved) publishes no binary for $($script:Target)"
		return $false
	}

	Write-Step 'method' "release binary  $($script:CDim)$($script:Target)$($script:CReset)"
	$archive = Join-Path $script:Temp $name
	Save-Url "$base/$name" $archive $name

	$sums = Join-Path $script:Temp 'SHA256SUMS'
	try {
		$req = New-Request "$base/SHA256SUMS"
		$resp = $req.GetResponse()
		$reader = New-Object System.IO.StreamReader($resp.GetResponseStream())
		[System.IO.File]::WriteAllText($sums, $reader.ReadToEnd())
		$reader.Dispose() ; $resp.Dispose()
	} catch {
		Stop-Install "release v$($script:Resolved) publishes no SHA256SUMS - refusing to install unverified"
	}

	$digest = Test-Checksum $archive $sums $name
	if (Test-Provenance $archive) {
		Write-Ok 'verified' "sha256 $digest  $($script:CDim)$([char]0x00B7)  provenance attested$($script:CReset)"
	} else {
		Write-Ok 'verified' "sha256 $digest"
	}

	# bsdtar has shipped in Windows since 10 build 17063 and reads .tar.gz
	# directly; nothing else in the box does.
	if (-not (Get-Command tar.exe -ErrorAction SilentlyContinue)) {
		Stop-Install 'tar.exe is missing - Windows 10 build 17063 or newer is required for the binary install'
	}
	$unpack = Join-Path $script:Temp 'x'
	New-Item -ItemType Directory -Force -Path $unpack | Out-Null
	& tar.exe -xzf $archive -C $unpack
	if ($LASTEXITCODE -ne 0) { Stop-Install "cannot unpack $name" }

	$exe = Join-Path $unpack $ExeName
	if (-not (Test-Path -LiteralPath $exe)) { Stop-Install "$name does not contain $ExeName" }
	Install-Binary $exe
	return $true
}

# ---------------------------------------------------------------------------
# Method: source
#
# The fallback while no Windows binary is published. The checkout stays in the
# cache directory so a later re-run is an incremental rebuild, not a cold one.
# ---------------------------------------------------------------------------

function Install-FromSource {
	Write-Step 'method' "source build  $($script:CDim)git + cargo$($script:CReset)"
	Write-Warn 'Windows source builds are not yet exercised upstream and may fail'

	if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
		Stop-Install "a source build needs git:`n   winget install --id Git.Git"
	}
	if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
		Stop-Install "a source build needs Rust 1.94 or newer:`n   winget install --id Rustlang.Rustup"
	}
	if (-not (Get-Command link.exe -ErrorAction SilentlyContinue) -and
		-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
		Write-Warn 'no MSVC linker on PATH - if the build fails, install the C++ build tools:'
		Write-Note 'winget install --id Microsoft.VisualStudio.2022.BuildTools'
	}

	$ref = 'main'
	if ($Version) {
		$ref = "v$($Version.TrimStart('v'))"
	} else {
		$latest = Resolve-LatestVersion
		if ($latest) { $ref = "v$latest" }
	}
	$script:Resolved = $ref.TrimStart('v')

	$dest = Join-Path $script:InstallDir $ExeName
	$have = Get-InstalledVersion $dest
	if (-not $Force -and $have -and $have -eq $script:Resolved) {
		$script:Installed = $dest
		Write-Ok 'up to date' "$dest  $($script:CDim)$($script:Resolved)$($script:CReset)"
		return
	}

	$src = Join-Path $env:LOCALAPPDATA "atomdrift\scan-src"
	if (Test-Path -LiteralPath (Join-Path $src '.git')) {
		Write-Step 'source' "updating $src"
		& git -C $src fetch --quiet --depth 1 origin $ref
		if ($LASTEXITCODE -ne 0) { Stop-Install "cannot fetch $ref" }
		& git -C $src checkout --quiet --force FETCH_HEAD
		if ($LASTEXITCODE -ne 0) { Stop-Install "cannot check out $ref" }
	} else {
		Write-Step 'source' "cloning $ref into $src"
		New-Item -ItemType Directory -Force -Path (Split-Path -Parent $src) | Out-Null
		& git clone --quiet --depth 1 --branch $ref "https://github.com/$Repo.git" $src
		if ($LASTEXITCODE -ne 0) { Stop-Install "cannot clone $Repo at $ref" }
	}

	Write-Step 'build' "cargo build --release  $($script:CDim)(the analysis stack is large - expect a long build)$($script:CReset)"
	Push-Location $src
	try {
		& cargo build --release --locked --bin $BinName
		if ($LASTEXITCODE -ne 0) { Stop-Install "build failed in $src" }
	} finally {
		Pop-Location
	}

	$built = Join-Path $src "target\release\$ExeName"
	if (-not (Test-Path -LiteralPath $built)) { Stop-Install "the build produced no $ExeName" }
	Install-Binary $built
	Write-Note "source checkout kept at $src - delete it to reclaim the space"
}

# ---------------------------------------------------------------------------
# Optional analysis tools
#
# None of these are required: scans work without them, with less depth on some
# file types. winget and scoop both install per-user without a UAC prompt, so
# they can be driven directly; anything else gets printed for the reader to run.
# ---------------------------------------------------------------------------

function Test-Tool([string[]]$Names) {
	foreach ($n in $Names) {
		if (Get-Command $n -ErrorAction SilentlyContinue) { return $true }
	}
	return $false
}

function Install-OptionalTools {
	if ($NoTools) { return }

	$scoop = [bool](Get-Command scoop -ErrorAction SilentlyContinue)
	$winget = [bool](Get-Command winget -ErrorAction SilentlyContinue)

	# name, commands that satisfy it, scoop package, winget id
	$tools = @(
		@{ Name = 'rizin'; Cmds = @('rizin', 'radare2', 'r2'); Scoop = 'rizin'; Winget = '' },
		@{ Name = 'upx'; Cmds = @('upx'); Scoop = 'upx'; Winget = 'UPX.UPX' },
		@{ Name = '7z'; Cmds = @('7z', '7zz'); Scoop = '7zip'; Winget = '7zip.7zip' },
		@{ Name = 'innoextract'; Cmds = @('innoextract'); Scoop = 'innoextract'; Winget = '' }
	)

	$report = @()
	$pending = @()
	foreach ($tool in $tools) {
		if (Test-Tool $tool.Cmds) {
			$report += "$($script:CGreen)$($script:GOk)$($script:CReset)$($tool.Name)"
			continue
		}

		$installed = $false
		if ($scoop -and $tool.Scoop) {
			# One package at a time: a name this bucket happens not to carry
			# must not take the others down with it.
			& scoop install $tool.Scoop *> $null
			$installed = Test-Tool $tool.Cmds
		} elseif ($winget -and $tool.Winget) {
			& winget install --id $tool.Winget --silent --accept-package-agreements --accept-source-agreements *> $null
			$installed = Test-Tool $tool.Cmds
		}

		if ($installed) {
			$report += "$($script:CGreen)$($script:GOk)$($script:CReset)$($tool.Name)"
		} else {
			$report += "$($script:CDim)-$($tool.Name)$($script:CReset)"
			if ($scoop -and $tool.Scoop) {
				$pending += "scoop install $($tool.Scoop)"
			} elseif ($tool.Winget) {
				$pending += "winget install --id $($tool.Winget)"
			} elseif ($tool.Name -eq 'rizin') {
				$pending += 'https://rizin.re/download/'
			}
		}
	}

	Write-Step 'tools' (($report -join ' ') + "  $($script:CDim)optional$($script:CReset)")
	foreach ($cmd in $pending) { Write-Note "for deeper analysis:  $cmd" }
}

# ---------------------------------------------------------------------------
# PATH
#
# The user PATH, never the machine one: this is a per-user install and editing
# the machine PATH would need administrator rights we deliberately never ask
# for. Adding it twice is the classic installer bug, so check first.
# ---------------------------------------------------------------------------

function Add-ToUserPath([string]$Directory) {
	$current = [Environment]::GetEnvironmentVariable('Path', 'User')
	if ($null -eq $current) { $current = '' }
	$parts = $current -split ';' | Where-Object { $_ -ne '' }
	foreach ($p in $parts) {
		if ($p.TrimEnd('\') -ieq $Directory.TrimEnd('\')) { return $false }
	}
	$updated = if ($current) { "$current;$Directory" } else { $Directory }
	[Environment]::SetEnvironmentVariable('Path', $updated, 'User')
	# Make it work in this session too, not only in the next one.
	$env:Path = "$env:Path;$Directory"
	return $true
}

function Write-Summary {
	if ($Quiet) { return }
	$version = Get-InstalledVersion $script:Installed
	if (-not $version) { $version = $script:Resolved }
	Write-Host ''
	Write-Host (" {0}{1}{2} {3}{4} {5}{6}  {7}{8}{9}" -f
		$script:CGreen, $script:GOk, $script:CReset, $script:CBold, $BinName, $version,
		$script:CReset, $script:CDim, $script:Installed, $script:CReset)
	Write-Host ''
	Write-Host ("   {0}scan a project{1}     {2} .\project" -f $script:CDim, $script:CReset, $BinName)
	Write-Host ("   {0}scan a package{1}     {2} purl npm/left-pad@1.3.0" -f $script:CDim, $script:CReset, $BinName)
	Write-Host ("   {0}everything else{1}    {2} --help" -f $script:CDim, $script:CReset, $BinName)
	Write-Host ''
	Write-Host ("   {0}The first scan downloads the model, rule, and bloom-filter bundles.{1}" -f
		$script:CDim, $script:CReset)
	Write-Host ''
}

# ---------------------------------------------------------------------------

function Invoke-Install {
	Initialize-Style
	Initialize-Tls
	Write-Banner
	Resolve-Platform
	Write-TargetList

	$script:Temp = Join-Path ([System.IO.Path]::GetTempPath()) "atomscan-install-$([guid]::NewGuid().ToString('N'))"
	New-Item -ItemType Directory -Force -Path $script:Temp | Out-Null
	try {
		Resolve-InstallDir

		$chosen = $Method
		if ($chosen -eq 'Auto') { $chosen = 'Binary' }
		if ($chosen -eq 'Binary') {
			if (-not (Install-FromRelease)) {
				Write-Note 'no binary available - building from source instead'
				$chosen = 'Source'
			}
		}
		if ($chosen -eq 'Source') { Install-FromSource }
	} finally {
		Remove-Item -LiteralPath $script:Temp -Recurse -Force -ErrorAction SilentlyContinue
	}

	if (-not $script:Installed) { Stop-Install 'the install produced no binary' }
	if (-not (Get-InstalledVersion $script:Installed)) {
		Stop-Install "$($script:Installed) does not run on this machine"
	}
	if ($script:Changed) { Write-Ok 'installed' $script:Installed }

	Install-OptionalTools

	if (-not $NoPath) {
		if (Add-ToUserPath (Split-Path -Parent $script:Installed)) {
			Write-Ok 'path' "added $(Split-Path -Parent $script:Installed) to your user PATH"
			Write-Note 'open a new terminal for it to take effect everywhere'
		}
	}

	$found = Get-Command $BinName -ErrorAction SilentlyContinue
	if ($found -and $found.Source -and $found.Source -ne $script:Installed) {
		Write-Warn "an earlier $BinName on your PATH will still win: $($found.Source)"
	}

	Write-Summary
}

Invoke-Install
