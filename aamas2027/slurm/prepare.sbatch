#!/bin/bash
#SBATCH --job-name=hbs-aamas-data
#SBATCH --partition=infer
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -euo pipefail
cd "$1"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
/home/glazkov/personality-twins-arr/vllmenv/bin/python -m aamas2027.pipeline \
  --old-root /home/glazkov/personality-twins-arr/LLM-PersonaBench \
  --output "$2" --manifest aamas2027/manifest.json
