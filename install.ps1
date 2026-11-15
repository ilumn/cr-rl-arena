# Contestant setup for cr-rl-arena on Windows.
#
# Installs everything needed to train an agent against the simulator and to
# connect it to the tournament server:
#   - Python 3.12 (if no Python 3.10+ is found) and Rust (via rustup), using winget
#   - a virtual environment in .venv at the repo root
#   - the simulator (cr_sim, compiled from gym/), numpy and websockets
#   - optionally PyTorch (CPU build) with -WithTorch
#
# Run from the repo root, in PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\install.ps1
#   powershell -ExecutionPolicy Bypass -File .\install.ps1 -WithTorch
#
# Safe to run again: it skips whatever is already installed and rebuilds the
# simulator (do that after every `git pull`).

param(
    [switch]$WithTorch,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv"

function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host ""; Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $cargo = Join-Path $env:USERPROFILE ".cargo\bin"
    $env:Path = "$cargo;$user;$machine"
}

function Have-Winget { return [bool](Get-Command winget -ErrorAction SilentlyContinue) }

# Returns the path of a real Python 3.10-3.13, or $null. Skips the Microsoft
# Store "python.exe" stub, which only opens the Store.
function Find-Python {
    $candidates = @()
    if ($Python) { $candidates += $Python }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in "3.12", "3.13", "3.11", "3.10") {
            try {
                $p = (& py "-$v" -c "import sys; print(sys.executable)" 2>$null)
                if ($LASTEXITCODE -eq 0 -and $p) { $candidates += $p.Trim() }
            } catch {}
        }
    }
    foreach ($v in "312", "313", "311", "310") {
        $candidates += Join-Path $env:LOCALAPPDATA "Programs\Python\Python$v\python.exe"
        $candidates += "C:\Program Files\Python$v\python.exe"
    }
    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath -and $onPath.Source -notlike "*WindowsApps*") { $candidates += $onPath.Source }

    foreach ($c in $candidates) {
        if (-not (Test-Path $c)) { continue }
        try {
            $ver = (& $c -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null)
            if ($LASTEXITCODE -ne 0 -or -not $ver) { continue }
            $minor = [int]($ver.Trim().Split(".")[1])
            if ($ver.Trim().StartsWith("3.") -and $minor -ge 10 -and $minor -le 13) { return $c }
        } catch {}
    }
    return $null
}

function Test-SmartAppControl {
    try { return ((Get-MpComputerStatus -ErrorAction Stop).SmartAppControlState -eq "On") }
    catch { return $false }
}

function Explain-SmartAppControl {
    Write-Host ""
    Write-Host "Windows Smart App Control is ON on this PC. It blocks the small helper" -ForegroundColor Yellow
    Write-Host "programs that Rust compiles while building the simulator, so the build" -ForegroundColor Yellow
    Write-Host "fails with 'An Application Control policy has blocked this file'." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  1. Ask the organizers for a prebuilt cr_sim wheel (.whl) and install it with:"
    Write-Host "       .\.venv\Scripts\python -m pip install <path-to-wheel>"
    Write-Host "  2. Build on another computer, or in WSL / Linux / macOS."
    Write-Host "  3. Turn Smart App Control off (Windows Security > App & browser control >"
    Write-Host "     Smart App Control). This is your decision: on some Windows versions it"
    Write-Host "     cannot be turned back on without reinstalling Windows."
}

Write-Host "cr-rl-arena contestant setup" -ForegroundColor Cyan
if (-not (Test-Path (Join-Path $Root "gym\Cargo.toml"))) {
    Fail "Run this script from the cr-rl-arena repo (gym\Cargo.toml not found next to it)."
}

# ---- 1. Python -------------------------------------------------------------
Step "Python 3.10+"
$py = Find-Python
if (-not $py) {
    if (-not (Have-Winget)) { Fail "No Python 3.10-3.13 found and winget is unavailable. Install Python 3.12 from https://www.python.org/downloads/ and run this again." }
    Warn "Not found. Installing Python 3.12 for your user account with winget..."
    winget install --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
    Refresh-Path
    $py = Find-Python
    if (-not $py) { Fail "Python was installed but could not be found. Open a new terminal and run this script again." }
}
Ok ("Using " + (& $py --version) + " at $py")

# ---- 2. Rust ---------------------------------------------------------------
Step "Rust toolchain"
Refresh-Path
if (-not (Get-Command rustup -ErrorAction SilentlyContinue)) {
    if (-not (Have-Winget)) { Fail "Rust is not installed and winget is unavailable. Install it from https://rustup.rs and run this again." }
    Warn "Not found. Installing rustup with winget..."
    winget install --id Rustlang.Rustup --silent --accept-package-agreements --accept-source-agreements
    Refresh-Path
    if (-not (Get-Command rustup -ErrorAction SilentlyContinue)) { Fail "rustup was installed but could not be found. Open a new terminal and run this script again." }
}

# The default (MSVC) toolchain needs Visual Studio Build Tools. Without them,
# use the GNU toolchain, which brings its own linker.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$hasMsvc = $false
if (Test-Path $vswhere) {
    $vc = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    $hasMsvc = [bool]$vc
}
if ($hasMsvc) {
    $toolchain = "stable-x86_64-pc-windows-msvc"
    Ok "Visual C++ Build Tools found; using the MSVC toolchain."
} else {
    $toolchain = "stable-x86_64-pc-windows-gnu"
    Ok "No Visual C++ Build Tools; using the GNU toolchain (no Visual Studio needed)."
}
rustup toolchain install $toolchain --profile minimal
if ($LASTEXITCODE -ne 0) { Fail "Could not install the Rust toolchain $toolchain." }
$env:RUSTUP_TOOLCHAIN = $toolchain   # this script only; your rustup default is unchanged
Ok (& rustc --version)

# ---- 3. Virtual environment and Python packages ----------------------------
Step "Virtual environment (.venv) and packages"
$vpy = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    & $py -m venv $Venv
    if ($LASTEXITCODE -ne 0) { Fail "Could not create the virtual environment." }
}
& $vpy -m pip install --quiet --upgrade pip
& $vpy -m pip install --quiet --upgrade "maturin>=1.7,<2.0" "numpy>=1.23" "websockets>=14" "pytest>=7"
if ($LASTEXITCODE -ne 0) { Fail "pip could not install maturin / numpy / websockets." }
if ($WithTorch) {
    Warn "Installing PyTorch (CPU build, about 200 MB)..."
    & $vpy -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { Fail "pip could not install PyTorch." }
}
Ok "Packages installed."

# ---- 4. Build the simulator ------------------------------------------------
Step "Building the simulator (first build takes a few minutes)"
$sac = Test-SmartAppControl
if ($sac) { Warn "Smart App Control is ON; the build may be blocked. Trying anyway..." }
$env:VIRTUAL_ENV = $Venv
# maturin refuses to run when VIRTUAL_ENV and CONDA_PREFIX are both set, which
# happens whenever a conda environment is active. Hide it for the build only.
$condaPrefix = $env:CONDA_PREFIX
if ($condaPrefix) { Warn "A conda environment is active; ignoring it for the build (.venv is used)." }
$env:CONDA_PREFIX = $null
Push-Location (Join-Path $Root "gym")
try {
    & (Join-Path $Venv "Scripts\maturin.exe") develop --release
    $buildOk = ($LASTEXITCODE -eq 0)
} finally {
    Pop-Location
    if ($condaPrefix) { $env:CONDA_PREFIX = $condaPrefix }
}
if (-not $buildOk) {
    if ($sac) { Explain-SmartAppControl }
    Fail "The simulator did not build. See the messages above."
}

# ---- 5. Check it works -----------------------------------------------------
Step "Checking the install"
& $vpy (Join-Path $Root "scripts\check_install.py")
if ($LASTEXITCODE -ne 0) { Fail "The simulator built but the check failed. See the messages above." }

Write-Host ""
Write-Host "All set." -ForegroundColor Green
Write-Host "  Activate the environment:  .\.venv\Scripts\Activate.ps1"
Write-Host "  Try the example agent:     python server\examples\random_agent.py ws://localhost:8765 my-team"
Write-Host "  Read next:                 gym\README.md and server\CONNECTING.md"
Write-Host "  After every git pull, run this script again to rebuild the simulator."
