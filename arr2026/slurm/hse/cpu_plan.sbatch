#!/bin/bash
#SBATCH --job-name=arr-cpu
#SBATCH --partition=rocky
#SBATCH --account=proj_1759
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=1-7
#SBATCH --output=/home/lsavchenko/personality-arr/logs/cpu_%A_%a.out
#SBATCH --error=/home/lsavchenko/personality-arr/logs/cpu_%A_%a.err

set -euo pipefail
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
module load Python/Anaconda 2>/dev/null || true
cd /home/lsavchenko/personality-arr/LLM-PersonaBench

case $SLURM_ARRAY_TASK_ID in
  1) S=arr2026/scripts/e1_cluster_validity.py;        A="--boot 40 --boot-sample 40000 --sil-sample 20000 --perm 200" ;;
  2) S=arr2026/scripts/e2_nonllm_baselines.py;        A="--extra-splits 20" ;;
  3) S=arr2026/scripts/hyp/h1_proper_scoring.py;      A="" ;;
  4) S=arr2026/scripts/hyp/h2_c2st_ablation.py;       A="" ;;
  5) S=arr2026/scripts/hyp/h3_orientation_audit.py;   A="" ;;
  6) S=arr2026/scripts/hyp/h5_variance_decomposition.py; A="" ;;
  7) S=arr2026/scripts/hyp/h6_prompt_ceiling.py;      A="" ;;
esac
echo "=== HSE task $SLURM_ARRAY_TASK_ID -> $S $A ==="
python3 $S $A
echo "DONE rc=$?"
