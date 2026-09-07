#!/bin/bash
#SBATCH --job-name=arr-h10
#SBATCH --partition=rocky
#SBATCH --account=proj_1759
#SBATCH --qos=normal
# NOTE: no --mem on cHARISMa; memory is not a schedulable TRES here.
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=/home/lsavchenko/personality-arr/logs/h10_%j.out
#SBATCH --error=/home/lsavchenko/personality-arr/logs/h10_%j.err
set -euo pipefail
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
module load Python/Anaconda 2>/dev/null || true
cd /home/lsavchenko/personality-arr/LLM-PersonaBench
PY=$(command -v python3)
"$PY" -c "import pandas, numpy, sklearn, scipy" 2>/dev/null || {
  echo "FATAL: scientific stack missing from $PY -- module load did not take"; exit 1; }
echo "using python: $PY"
"$PY" arr2026/scripts/hyp/h10_stability_null.py --boot 30 --sub 30000
echo "DONE rc=$?"
