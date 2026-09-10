#!/bin/bash
# Build a venv that can load current-generation architectures. hfenv has
# transformers 4.49, which cannot build qwen3_5, granite-4.2, gemma4_unified,
# apertus1p5 or mistral3. Login node has internet; compute nodes do not.
set -euo pipefail
cd /home/lsavchenko/personality-arr
UV=/home/lsavchenko/.local/bin/uv
"$UV" venv --python 3.12 arrenv 2>&1 | tail -2
PY=/home/lsavchenko/personality-arr/arrenv/bin/python
# cu128 wheels cover sm_90 (H100/H200), which is all we target here.
"$UV" pip install --python "$PY" --quiet \
  --index-url https://download.pytorch.org/whl/cu128 torch 2>&1 | tail -3
"$UV" pip install --python "$PY" --quiet \
  "transformers>=5.16" accelerate safetensors sentencepiece \
  numpy pandas xlrd openpyxl huggingface_hub hf_transfer 2>&1 | tail -3
"$PY" - <<'PYEOF'
import importlib.metadata as m
for p in ("torch","transformers","accelerate","numpy","pandas"):
    try: print(f"{p:14s} {m.version(p)}")
    except Exception: print(f"{p:14s} ABSENT")
PYEOF
echo "ENV BUILD DONE"
