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
cd /home/lsavchenko/personality-arr/LLM-PersonaBench
source arr2026/slurm/hse/_pyresolve.sh
PY=$(resolve_py "pandas,numpy,sklearn,scipy") || { echo "FATAL: no python with the scientific stack"; exit 1; }
echo "using python: $PY"
"$PY" arr2026/scripts/hyp/x1_cross_instrument.py --n-boot 500 --n-eval 600
echo "DONE rc=$?"
