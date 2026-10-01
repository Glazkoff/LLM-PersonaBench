#!/bin/bash
#SBATCH --job-name=hbs-icml-preflight
#SBATCH --partition=infer
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/icml_preflight_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/icml_preflight_%j.err
# Gate G0 on Euler: unit tests, orientation regression, and validation of saved
# per-model GPU readout probe receipts. This job is CPU-only.
# usage: sbatch icml2027/slurm/preflight.sbatch [tier]
set -uo pipefail
export PATH=/home/glazkov/personality-twins-arr/vllmenv/bin:$PATH
export HF_HOME=/home/glazkov/hf_cache
export TOKENIZERS_PARALLELISM=false
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
TIER="${1:-1}"

echo "########## tests ##########"
"$PY" -m pytest tests/optim -q 2>&1 | tail -20
rc_tests=${PIPESTATUS[0]}

echo "########## orientation regression on evoprompt_iter2 artefacts ##########"
"$PY" -m src.optim.checks orientation_regression --min-corr 0.60 2>&1 | tail -5
rc_orient=${PIPESTATUS[0]}

echo "########## saved readout probe receipts, tier $TIER ##########"
# Probe inference was already run inside finite GPU Slurm jobs. Validate its receipts here.
"$PY" -m src.optim.checks readout_guard --models-yaml icml2027/configs/models.yaml --tier "$TIER" \
  --out icml2027/results/aggregates/preflight_readout.json 2>&1 | tail -30
rc_guard=${PIPESTATUS[0]}

echo "PREFLIGHT tests=$rc_tests orientation=$rc_orient guard=$rc_guard"
[ "$rc_tests" = 0 ] && [ "$rc_orient" = 0 ] && [ "$rc_guard" = 0 ]
