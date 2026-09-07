#!/bin/bash
#SBATCH --job-name=arr-x1
#SBATCH --partition=rocky
#SBATCH --account=proj_1759
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/home/lsavchenko/personality-arr/logs/x1_%j.out
#SBATCH --error=/home/lsavchenko/personality-arr/logs/x1_%j.err
set -euo pipefail
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
module load Python/Anaconda 2>/dev/null || true
cd /home/lsavchenko/personality-arr/LLM-PersonaBench
PY=$(command -v python3)
"$PY" -c "import pandas,numpy" 2>/dev/null || { echo "FATAL: stack missing"; exit 1; }
"$PY" arr2026/scripts/hyp/x1_cross_instrument.py --n-boot 500 --n-eval 600
echo "DONE rc=$?"
