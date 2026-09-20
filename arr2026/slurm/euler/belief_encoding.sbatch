#!/bin/bash
#SBATCH --job-name=hbs-enc
#SBATCH --partition=infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/enc_%A_%a.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/enc_%A_%a.err
# Interventional counterpart to the h11 ceiling: keep the model, the personas and
# every other word of the prompt fixed, and widen ONLY the channel that carries
# who the respondent is (5 coarse words -> the number itself). If individuation
# is capped by the encoding rather than by the model, VR_between must rise.
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/glazkov/hf_cache
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench

case $SLURM_ARRAY_TASK_ID in
  1) MODEL="ibm-granite/granite-4.1-8b";               TAG=granite_8b;  BS=8 ;;
  2) MODEL="swiss-ai/Apertus-8B-Instruct-2509";        TAG=apertus_8b;  BS=8 ;;
  3) MODEL="openai/gpt-oss-20b";                       TAG=gptoss_20b;  BS=4 ;;
esac
ENC=${ENC:-raw}

PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
echo "=== $MODEL | encoding=$ENC on $(hostname) ==="
"$PY" arr2026/scripts/hyp/h4_variance_ladder_hf.py \
  --model "$MODEL" --clusters 0 1 2 3 --n-personas 40 --encoding "$ENC" \
  --batch-size "$BS" --out "arr2026/results_euler/h4_${TAG}_${ENC}"
echo "DONE rc=$?"
