#!/usr/bin/env bash
set -euo pipefail

# 0) Make sure we're in the script's directory, not wherever you happened to call it from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1) Prerequisites: python3 and the venv module
if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not found. Install it with:"
  echo "  sudo apt update && sudo apt install python3"
  exit 1
fi

if ! python3 -c 'import venv' &>/dev/null; then
  echo "ERROR: python3-venv module is missing. Install it with:"
  echo "  sudo apt update && sudo apt install python3-venv"
  exit 1
fi

# 2) Create the venv if it doesn't already exist
if [[ ! -d "venv" ]]; then
  echo "[+] Creating virtualenv in ./venv"
  python3 -m venv venv
else
  echo "[i] venv/ already exists, skipping creation"
fi

# 3) Verify that the activate script actually got created
if [[ ! -f "venv/bin/activate" ]]; then
  echo "ERROR: venv/bin/activate not found!"
  echo "Directory listing of venv/bin:"
  ls -l venv/bin
  exit 1
fi

# 4) Activate it (in this shell or subshell)
echo "[+] Activating venv"
# shellcheck disable=SC1091
source venv/bin/activate

# 5) Now you can do whatever comes next…
#    e.g. upgrade pip, install requirements, etc.
echo "[+] Upgrading pip & installing your packages"
python -m pip install --upgrade pip
python -m pip install numpy cvxpy ecos scs matplotlib imageio pandas

echo "✅ Done! venv is ready."
echo "   To use it in _this_ shell: you’re already in it."
echo "   To use it in a fresh shell, run:  source $SCRIPT_DIR/venv/bin/activate"
