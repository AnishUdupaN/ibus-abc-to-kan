#!/usr/bin/env bash
set -euo pipefail

echo "[container] Installing uv..."
dnf install -y -q curl gcc gcc-c++
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"

echo "[container] Creating Python 3.10 export venv..."
mkdir -p /build /out
uv venv /build/export-venv --python 3.10

echo "[container] Installing CPU-only PyTorch + IndicXlit + ONNX..."
uv pip install --python /build/export-venv/bin/python \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    torch ai4bharat-transliteration "onnxruntime<1.20" onnx "numpy<2"

echo "[container] Running ONNX & SQLite export..."
OUT_DIR=/out /build/export-venv/bin/python /work/export_kn_onnx.py
