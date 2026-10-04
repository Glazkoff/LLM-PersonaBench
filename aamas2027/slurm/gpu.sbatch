#!/bin/bash
#SBATCH --job-name=hbs-aamas-evo
#SBATCH --partition=infer
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h200:2
set -euo pipefail
CAMPAIGN="$1"
CODE="$2"
PHASE="$3"
INDEX="${4:-${SLURM_ARRAY_TASK_ID}}"
cd "$CODE"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export HBS_GIGAEVO_SOURCE="$CAMPAIGN/gigaevo-f72ab08"
export HBS_GIGAEVO_PYTHON="$CAMPAIGN/native-env/bin/python"
sha256sum --check SOURCE_SHA256.txt > "$CAMPAIGN/logs/source-check-${SLURM_JOB_ID}.txt"
"$CAMPAIGN/env/bin/python" -m aamas2027.serve "$PHASE" --index "$INDEX" --manifest aamas2027/manifest.json --results "$CAMPAIGN/results-v3"
