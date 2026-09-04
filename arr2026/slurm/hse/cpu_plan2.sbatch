#!/bin/bash
#SBATCH --job-name=arr-cpu2
#SBATCH --partition=rocky
#SBATCH --account=proj_1759
#SBATCH --qos=normal
# NOTE: no --mem on cHARISMa; memory is not a schedulable TRES here.
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=7-9
#SBATCH --output=/home/lsavchenko/personality-arr/logs/cpu2_%A_%a.out
#SBATCH --error=/home/lsavchenko/personality-arr/logs/cpu2_%A_%a.err

set -euo pipefail
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
module load Python/Anaconda 2>/dev/null || true
cd /home/lsavchenko/personality-arr/LLM-PersonaBench

case $SLURM_ARRAY_TASK_ID in
  7) S=arr2026/scripts/hyp/h7_permutation_null.py ;;
  8) S=arr2026/scripts/hyp/h8_respondent_equivalents.py ;;
  9) S=arr2026/scripts/hyp/h9_metric_sample_size.py ;;
esac
echo "=== HSE task $SLURM_ARRAY_TASK_ID -> $S ==="
python3 $S
echo "DONE rc=$?"
