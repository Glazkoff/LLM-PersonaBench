#!/bin/bash
#SBATCH --job-name=arr-h2
#SBATCH --partition=infer
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/h2_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/h2_%j.err
set -euo pipefail
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
"$PY" -c "import sklearn" 2>/dev/null || /home/glazkov/.local/bin/uv pip install --python "$PY" --quiet scikit-learn
"$PY" arr2026/scripts/hyp/h2_c2st_ablation.py 2>&1 | tail -30
echo "DONE rc=$?"
