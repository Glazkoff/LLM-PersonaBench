#!/bin/bash
#SBATCH --job-name=hbs-probe
#SBATCH --partition=infer
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/probe_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/probe_%j.err
# CPU-only, but a Slurm job all the same: nothing runs bare on this cluster.
set -euo pipefail
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python

echo "=== installing the two missing deps into vllmenv ==="
# vllmenv is a uv venv and has no pip of its own.
UV=/home/glazkov/.local/bin/uv
"$UV" pip install --python "$PY" --quiet pandas accelerate 2>&1 | tail -3
"$PY" -c "import pandas, accelerate, transformers, torch; \
  print('pandas', pandas.__version__, '| accelerate', accelerate.__version__, \
        '| tf', transformers.__version__, '| torch', torch.__version__)"

echo "=== probing candidate families ==="
export HF_HOME=/home/glazkov/hf_cache
"$PY" arr2026/scripts/hyp/probe_models.py --per-family 2 || true
echo "DONE rc=$?"
