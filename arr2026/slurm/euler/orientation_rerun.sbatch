#!/bin/bash
#SBATCH --job-name=hbs-orient
#SBATCH --partition=infer
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/orient_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/orient_%j.err
# Regenerate every analysis that compares model answers to human answers, now
# that all of them share one orientation transform. CPU-only, but a Slurm job.
set -euo pipefail
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
"$PY" -c "import sklearn" 2>/dev/null || /home/glazkov/.local/bin/uv pip install --python "$PY" --quiet scikit-learn
for s in h1_proper_scoring h5_variance_decomposition h7_permutation_null; do
  echo "=============== $s ==============="
  "$PY" "arr2026/scripts/hyp/$s.py" 2>&1 | tail -25 || echo "  ($s FAILED)"
done
echo "=============== e3_metric_suite ==============="
"$PY" arr2026/scripts/e3_metric_suite.py 2>&1 | tail -20 || echo "  (e3 FAILED)"
echo "DONE rc=$?"
