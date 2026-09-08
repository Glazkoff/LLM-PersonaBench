#!/bin/bash
#SBATCH --job-name=arr-h11
#SBATCH --partition=infer
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/h11_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/h11_%j.err
# CPU-only, but a Slurm job all the same: nothing runs bare on this cluster.
set -euo pipefail
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
"$PY" -c "import sklearn" 2>/dev/null \
  || /home/glazkov/.local/bin/uv pip install --python "$PY" --quiet scikit-learn
"$PY" arr2026/scripts/hyp/h11_prompt_channel_ceiling.py --boost \
  --out arr2026/results_euler/h11_prompt_ceiling
echo "DONE rc=$?"
