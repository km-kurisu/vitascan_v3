#!/usr/bin/env bash
# Start the VitaScan backend and frontend together (Linux / macOS).
# All arguments are passed through: ./start.sh --prod, ./start.sh --setup
set -euo pipefail

cd "$(dirname "$0")"

pick_python() {
  for candidate in backend/venv/bin/python python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

if ! PYTHON="$(pick_python)"; then
  echo "error | Python 3.12+ not found. Install it and re-run ./start.sh" >&2
  exit 1
fi

exec "$PYTHON" start.py "$@"
