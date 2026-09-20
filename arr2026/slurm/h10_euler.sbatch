#!/bin/bash
#SBATCH --job-name=hbs-h10
#SBATCH --partition=infer
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/h10_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/h10_%j.err
set -euo pipefail
export OMP_NUM_THREADS=48 OPENBLAS_NUM_THREADS=48 MKL_NUM_THREADS=48
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
/home/glazkov/.local/bin/uv run --with pandas --with numpy --with scipy --with scikit-learn \
  python arr2026/scripts/hyp/h10_stability_null.py --boot 30 --sub 30000
echo "DONE rc=$?"
