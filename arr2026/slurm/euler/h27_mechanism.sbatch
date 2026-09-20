#!/bin/bash
#SBATCH --job-name=hbs-h27
#SBATCH --partition=infer
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/h27_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/h27_%j.err
# H27: what determines a persona's answer-equivalent? CPU-only, no --gres.
#
# Needs the belief tensors themselves, which live only on this box, to measure
# between-person forecast dispersion WITHOUT fitting anything -- the
# non-circular version of the mechanism test. alpha* is fitted against the same
# loss m* is derived from, so it cannot carry the mechanism claim on its own.
set -uo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
"$PY" arr2026/scripts/hyp/h27_mechanism.py \
  --equiv "arr2026/results_euler/h26_equiv/*.json" \
  --runs-root arr2026/results_euler \
  --ladders arr2026/results_euler \
  --out arr2026/results_euler/h27_mechanism.json
rc=$?; echo "DONE rc=$rc"; exit $rc
