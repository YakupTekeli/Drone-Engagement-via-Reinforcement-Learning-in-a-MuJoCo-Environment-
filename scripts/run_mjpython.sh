#!/bin/sh
# Proje yolundaki boşluk nedeniyle bozulan venv console-script shebang'ini atlar.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(dirname -- "$SCRIPT_DIR")
exec "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/.venv/bin/mjpython" "$@"
