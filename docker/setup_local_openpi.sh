#!/bin/bash
set -euo pipefail

source /opt/venv/openpi/bin/activate

echo "[OpenPI override] removing packaged OpenPI..."
uv pip uninstall rlinf-openpi openpi || true

echo "[OpenPI override] installing local OpenPI..."
uv pip install \
    --no-config \
    --no-deps \
    -e /opt/openpi-lerobot

echo "[OpenPI override] installing LeRobot V3 runtime..."
uv pip install \
    --no-config \
    --no-deps \
    "lerobot==0.4.4" \
    "datasets==4.0.0"

python - <<'PY'
import torch
import openpi
import lerobot
import datasets

print("torch:", torch.__version__)
print("openpi:", openpi.__file__)
print("lerobot:", lerobot.__version__)
print("datasets:", datasets.__version__)

assert lerobot.__version__ == "0.4.4"
assert datasets.__version__ == "4.0.0"
assert openpi.__file__.startswith("/opt/openpi-lerobot/")
PY