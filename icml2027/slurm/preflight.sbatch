#!/bin/bash
#SBATCH --job-name=icml-preflight
#SBATCH --partition=infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/icml_preflight_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/icml_preflight_%j.err
# Gate G0 on Euler: unit tests, orientation regression, and the readout guard per Tier-1 model.
# usage: sbatch icml2027/slurm/preflight.sbatch [tier]
set -uo pipefail
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

echo "########## readout guard, tier $TIER ##########"
# For each model: start vLLM, read 8 respondents x 20 items, report mean mass on scale, stop.
"$PY" -m src.optim.checks readout_guard --models-yaml icml2027/configs/models.yaml --tier "$TIER" \
  --n-respondents 8 --n-items 20 --out icml2027/results/aggregates/preflight_readout.json 2>&1 | tail -30
rc_guard=${PIPESTATUS[0]}

echo "PREFLIGHT tests=$rc_tests orientation=$rc_orient guard=$rc_guard"
[ "$rc_tests" = 0 ] && [ "$rc_orient" = 0 ] && [ "$rc_guard" = 0 ]
