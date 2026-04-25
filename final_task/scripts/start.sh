#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG_PATH="${1:-config/defaults.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
COUNTRIES="${COUNTRIES:-}"

echo "[start] Installing Python dependencies"
"$PYTHON_BIN" -m pip install --upgrade pip >/dev/null
"$PYTHON_BIN" -m pip install -r requirements.txt

echo "[start] Launching mission with config: $CONFIG_PATH"
if [ -n "$COUNTRIES" ]; then
  "$PYTHON_BIN" -m src.mission --config "$CONFIG_PATH" --countries "$COUNTRIES"
else
  "$PYTHON_BIN" -m src.mission --config "$CONFIG_PATH"
fi
