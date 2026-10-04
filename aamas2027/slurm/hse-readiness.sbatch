#!/bin/bash
#SBATCH --job-name=hbs-aamas-hse-check
#SBATCH --partition=rocky
#SBATCH --qos=normal
#SBATCH --account=proj_1759
#SBATCH --constraint=type_a|type_b
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --time=00:05:00
set -euo pipefail
ROOT="$1"
MODE="$2"
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ARGS=()
if test "$MODE" = gpu; then ARGS+=(--gpu); fi
/home/lsavchenko/personality-arr/icmlenv/bin/python "$ROOT/hse_readiness.py" \
  --lock "$ROOT/model_lock.json" --output "$ROOT/readiness-${MODE}-${SLURM_JOB_ID}.json" "${ARGS[@]}"
