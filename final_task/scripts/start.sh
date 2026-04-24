#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG_PATH="${1:-config/defaults.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
COUNTRIES="${COUNTRIES:-}"

if [ ! -d ".venv" ]; then
  echo "[start] Creating virtual environment"
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
. .venv/bin/activate

echo "[start] Installing Python dependencies"
python3 -m pip install --upgrade pip >/dev/null
python3 -m pip install -r requirements.txt

echo "[start] Launching mission with config: $CONFIG_PATH"
if [ -n "$COUNTRIES" ]; then
  python3 -m src.mission --config "$CONFIG_PATH" --countries "$COUNTRIES"
else
  python3 -m src.mission --config "$CONFIG_PATH"
fi
