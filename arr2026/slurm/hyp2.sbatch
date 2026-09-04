#!/bin/bash
#SBATCH --job-name=arr-hyp2
#SBATCH --partition=infer
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=5-6
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/hyp2_%A_%a.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/hyp2_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
case $SLURM_ARRAY_TASK_ID in
  5) SCRIPT=arr2026/scripts/hyp/h5_variance_decomposition.py ;;
  6) SCRIPT=arr2026/scripts/hyp/h6_prompt_ceiling.py ;;
esac
echo "=== task $SLURM_ARRAY_TASK_ID -> $SCRIPT ==="
/home/glazkov/.local/bin/uv run --with pandas --with numpy --with scipy \
  --with scikit-learn --with xlrd --with openpyxl python "$SCRIPT"
echo "DONE rc=$?"
