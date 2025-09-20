#!/usr/bin/env bash
# setup_project_env.sh — bootstrap a project-local Python venv on macOS (Intel)
set -euo pipefail
trap 'echo "❌ Setup failed at line $LINENO" >&2' ERR

log() { printf "\n\033[1m%s\033[0m\n" "$*"; }
run() { echo "+ $*"; eval "$@"; }

PROJECT_DIR="$(pwd)"

log "🔎 Checking prerequisites"

# 0) (Optional) Xcode Command Line Tools — not strictly required but helpful
if ! xcode-select -p >/dev/null 2>&1; then
  echo "• Command Line Tools not found. You may be prompted to install them."
  # This opens an Apple dialog; continue if user cancels.
  xcode-select --install || true
fi

# 1) Homebrew
if ! command -v brew >/dev/null 2>&1; then
  log "🍺 Installing Homebrew (not found)"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Ensure Homebrew is available in this script's PATH
if ! command -v brew >/dev/null 2>&1; then
  if [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  elif [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  fi
fi

log "✅ Homebrew ready: $(brew --version | head -n1)"

# 2) Python 3 (via Homebrew)
if ! command -v python3 >/dev/null 2>&1; then
  log "🐍 Installing Python 3"
  run brew install python
fi

PYTHON_BIN="$(command -v python3)"
log "Using python3 at: $PYTHON_BIN"
echo "Version: $("$PYTHON_BIN" -V)"

# 3) Create .venv (project-local)
if [[ ! -d ".venv" ]]; then
  log "📦 Creating virtual environment: .venv"
  run "$PYTHON_BIN" -m venv .venv
else
  log "ℹ️  .venv already exists — reusing it"
fi

VENV_PY="./.venv/bin/python"
VENV_PIP="./.venv/bin/pip"

# 4) Upgrade pip inside venv
log "⬆️  Upgrading pip in venv"
run "$VENV_PY" -m pip install --upgrade pip

# 5) Install dependencies if present
if [[ -f "requirements.txt" ]]; then
  log "📥 Installing dependencies from requirements.txt"
  run "$VENV_PIP" install -r requirements.txt
else
  log "⚠️  No requirements.txt found — skipping dependency install"
fi

# 6) Freeze a lock snapshot for reproducibility
log "📌 Writing dependency snapshot -> requirements.lock.txt"
run "$VENV_PIP" freeze > requirements.lock.txt

# 7) Generate a safe activation helper for zsh/bash (to source manually)
log "🧩 Creating activation helper: ./activate.zsh"
cat > activate.zsh <<'EOF'
# Source this file to activate the project's virtualenv.
# Usage:
#   source ./activate.zsh     # zsh or bash
if [[ -f ".venv/bin/activate" ]]; then
  . ".venv/bin/activate"
  echo "✅ .venv activated. Run 'deactivate' to exit."
else
  echo "❌ .venv not found. Create it with: python3 -m venv .venv" >&2
fi
EOF
chmod +x activate.zsh

# 8) Minimal quickstart readme
log "📝 Writing ENV-QUICKSTART.md"
cat > ENV-QUICKSTART.md <<'EOF'
# Project Environment Quickstart

## Create / update the environment
```sh
bash ./setup_project_env.sh
