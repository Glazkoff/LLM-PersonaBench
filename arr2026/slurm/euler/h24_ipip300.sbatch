#!/bin/bash
#SBATCH --job-name=hbs-h24t
#SBATCH --partition=infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=1-8%2
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/h24_%A_%a.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/h24_%A_%a.err
# H24: same-person transfer to an unseen item family (IPIP-NEO-300).
#
# Conditions on the 120 short-form items every other experiment uses and reads
# out the 180 items the same person answered that no arm has been fitted on.
# This is the only genuine same-person unseen-item-family set in the data; see
# arr2026/strengthen/BRIEF.md section 13.
#
# 150 respondents x 180 items = 27,000 prompts per task, ~1.8x the h15 workload,
# which fitted comfortably in 8h on one GPU. 12h carries the history-mode tasks,
# whose prompts are far longer.
#
# %2 caps the array at two concurrent GPUs. `infer` is shared with medevo-* and
# hfr-*, and eight simultaneous H200 holds would starve them.
set -uo pipefail
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/glazkov/hf_cache
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
"$PY" -c "import xlrd" 2>/dev/null || \
  /home/glazkov/.local/bin/uv pip install --python "$PY" --quiet xlrd

# Model choice is driven by H26, not by convenience. The answer-equivalent ladder
# (BRIEF.md section 16.2) puts Qwen3.6-35B-A3B at m*=3.07 and Qwen3.8-27B at 1.31,
# while Qwen2.5-7B and granite-4.1-8b sit at 0.13 and 0.04. Running the transfer
# test on a model worth 0.04 answers would measure nothing, so the array targets
# the top of the ladder and keeps ONE weak model as the contrast arm.
case $SLURM_ARRAY_TASK_ID in
  1) M="Qwen/Qwen3.6-35B-A3B";       MODE=scores;  OBS=0;   TAG=qwen36_35b ;;
  2) M="Qwen/Qwen3.8-27B";           MODE=scores;  OBS=0;   TAG=qwen38_27b ;;
  3) M="ibm-granite/granite-4.2-30b";MODE=scores;  OBS=0;   TAG=granite42_30b ;;
  4) M="Qwen/Qwen2.5-7B-Instruct";   MODE=scores;  OBS=0;   TAG=qwen25_7b ;;
  5) M="Qwen/Qwen3.6-35B-A3B";       MODE=history; OBS=5;   TAG=qwen36_35b ;;
  6) M="Qwen/Qwen3.6-35B-A3B";       MODE=history; OBS=20;  TAG=qwen36_35b ;;
  7) M="Qwen/Qwen3.6-35B-A3B";       MODE=history; OBS=60;  TAG=qwen36_35b ;;
  8) M="Qwen/Qwen3.6-35B-A3B";       MODE=history; OBS=120; TAG=qwen36_35b ;;
  *) echo "no such task"; exit 2 ;;
esac
if [ "$MODE" = scores ]; then OUT="arr2026/results_euler/h24_${TAG}_scores"
else OUT="arr2026/results_euler/h24_${TAG}_hist${OBS}"; fi

echo "=== task $SLURM_ARRAY_TASK_ID: $M / $MODE obs=$OBS on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" arr2026/scripts/hyp/h24_ipip300_readout.py \
  --model "$M" --persona "$MODE" --n-observed "$OBS" \
  --n-scored 150 --batch-size ${BS:-4} --out "$OUT"
rc=$?
echo "DONE rc=$rc"
exit $rc
