#!/bin/bash
#SBATCH --job-name=arr-h21pred
#SBATCH --partition=infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/h21p_%A_%a.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/h21p_%A_%a.err
#
# PRE-REGISTERED PREDICTIONS, recorded before either run starts.
#
# Wave 1 produced the hypothesis that H21 fires REACHABLE exactly at the
# "near-human total spread, almost no individuation" corner: the only two
# REACHABLE models so far, gpt-oss-20b and Olmo-3-7B, are statistically
# indistinguishable in share_between (-0.0001 [-0.0017,+0.0016]) and both sit
# at ~0.016 with VR_total ~1.1. Six models elsewhere were NOT SEPARATED.
#
# These two models were measured AFTER that hypothesis was formed, so their H21
# outcome is a test of it rather than a fit to it:
#
#   Apertus-v1.1-4B-Instruct  share_between 0.0051, VR_total 1.025
#       -> deepest into the corner of any model measured. PREDICT REACHABLE.
#   Meta-Llama-3.1-8B-Instruct share_between 0.0790, VR_total 0.874
#       -> mid-range, away from the corner.       PREDICT NOT SEPARATED.
#
# A miss on either is evidence against the hypothesis and must be reported as
# such; the paper should not carry the mechanism if this fails.
set -uo pipefail
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/glazkov/hf_cache
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
MODELS=("swiss-ai/Apertus-v1.1-4B-Instruct:8" "unsloth/Meta-Llama-3.1-8B-Instruct:8")
E="${MODELS[$((SLURM_ARRAY_TASK_ID - 1))]}"; M="${E%:*}"; BS="${E##*:}"
TAG=$(echo "$M" | tr '/.' '__' | tr -d ' ')
echo "=== H21-pred $M (bs=$BS) ==="
"$PY" arr2026/scripts/hyp/h21_reachability.py --model "$M" --per-cluster 48 \
  --batch-size "$BS" --out "arr2026/results_euler/h21_${TAG}"
rc=$?; echo "DONE rc=$rc"; exit $rc
