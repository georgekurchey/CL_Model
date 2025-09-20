# Source this file to activate the project's virtualenv.
# Usage:
#   source ./activate.zsh     # zsh or bash
if [[ -f ".venv/bin/activate" ]]; then
  . ".venv/bin/activate"
  echo "✅ .venv activated. Run 'deactivate' to exit."
else
  echo "❌ .venv not found. Create it with: python3 -m venv .venv" >&2
fi
