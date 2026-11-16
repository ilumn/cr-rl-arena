#!/usr/bin/env bash
# Contestant setup for cr-rl-arena on macOS and Linux.
#
# Installs everything needed to train an agent against the simulator and to
# connect it to the tournament server:
#   - Rust (via rustup, into your home directory) if it is missing
#   - a virtual environment in .venv at the repo root
#   - the simulator (cr_sim, compiled from gym/), numpy and websockets
#   - optionally PyTorch with --torch
# Python 3.10+ and a C compiler must already be installed; the script says how
# to get them if they are not.
#
# Run from the repo root:
#   bash install.sh            # or: bash install.sh --torch
#
# Safe to run again: it skips whatever is already installed and rebuilds the
# simulator (do that after every `git pull`).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
WITH_TORCH=0
for arg in "$@"; do
  case "$arg" in
    --torch) WITH_TORCH=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

step() { printf '\n\033[36m==> %s\033[0m\n' "$1"; }
ok()   { printf '    \033[32m%s\033[0m\n' "$1"; }
warn() { printf '    \033[33m%s\033[0m\n' "$1"; }
fail() { printf '\n\033[31mERROR: %s\033[0m\n' "$1"; exit 1; }

echo "cr-rl-arena contestant setup"
[ -f "$ROOT/gym/Cargo.toml" ] || fail "Run this script from the cr-rl-arena repo (gym/Cargo.toml not found next to it)."
OS="$(uname -s)"

# ---- 1. Python ---------------------------------------------------------------
step "Python 3.10+"
PY=""
for c in python3.12 python3.13 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 &&
     "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; then
    PY="$(command -v "$c")"; break
  fi
done
if [ -z "$PY" ]; then
  if [ "$OS" = "Darwin" ]; then
    fail "No Python 3.10-3.13 found. Install one with 'brew install python@3.12' (or from python.org) and run this again."
  else
    fail "No Python 3.10-3.13 found. Install one (Ubuntu/Debian: 'sudo apt install python3 python3-venv') and run this again."
  fi
fi
ok "Using $("$PY" --version) at $PY"
if ! "$PY" -c 'import venv, ensurepip' >/dev/null 2>&1; then
  fail "Python's venv module is missing. On Ubuntu/Debian: 'sudo apt install python3-venv' (match your Python version, e.g. python3.12-venv)."
fi

# ---- 2. C compiler (Rust needs one to link) ----------------------------------
step "C compiler"
if [ "$OS" = "Darwin" ]; then
  xcode-select -p >/dev/null 2>&1 || fail "Xcode command line tools are missing. Run 'xcode-select --install', then run this again."
else
  command -v cc >/dev/null 2>&1 || fail "No C compiler found. Ubuntu/Debian: 'sudo apt install build-essential'. Fedora: 'sudo dnf install gcc'."
fi
ok "found"

# ---- 3. Rust -----------------------------------------------------------------
step "Rust toolchain"
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
if ! command -v cargo >/dev/null 2>&1; then
  warn "Not found. Installing rustup into ~/.cargo ..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  . "$HOME/.cargo/env"
fi
ok "$(rustc --version)"

# ---- 4. Virtual environment and Python packages ------------------------------
step "Virtual environment (.venv) and packages"
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
VPY="$VENV/bin/python"
"$VPY" -m pip install --quiet --upgrade pip
"$VPY" -m pip install --quiet --upgrade "maturin>=1.7,<2.0" "numpy>=1.23" "websockets>=14" "pytest>=7"
if [ "$WITH_TORCH" = 1 ]; then
  warn "Installing PyTorch..."
  if [ "$OS" = "Linux" ]; then
    # CPU build: far smaller than the default CUDA one. Drop the index URL for GPU.
    "$VPY" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
  else
    "$VPY" -m pip install --quiet torch
  fi
fi
ok "Packages installed."

# ---- 5. Build the simulator --------------------------------------------------
step "Building the simulator (first build takes a few minutes)"
# maturin refuses to run when VIRTUAL_ENV and CONDA_PREFIX are both set, which
# happens whenever a conda environment is active. Hide it for the build only.
[ -n "${CONDA_PREFIX:-}" ] && warn "A conda environment is active; ignoring it for the build (.venv is used)."
( cd "$ROOT/gym" && unset CONDA_PREFIX && VIRTUAL_ENV="$VENV" "$VENV/bin/maturin" develop --release ) \
  || fail "The simulator did not build. See the messages above."

# ---- 6. Check it works -------------------------------------------------------
step "Checking the install"
"$VPY" "$ROOT/scripts/check_install.py" || fail "The simulator built but the check failed. See the messages above."

printf '\n\033[32mAll set.\033[0m\n'
echo "  Activate the environment:  source .venv/bin/activate"
echo "  Try the example agent:     python server/examples/random_agent.py ws://localhost:8765 my-team"
echo "  Read next:                 gym/README.md and server/CONNECTING.md"
echo "  After every git pull, run this script again to rebuild the simulator."
