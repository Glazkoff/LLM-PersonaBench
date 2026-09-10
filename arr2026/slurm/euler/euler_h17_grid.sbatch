#!/bin/bash
#SBATCH --job-name=arr-h17
#SBATCH --partition=infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/h17_%A_%a.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/h17_%A_%a.err
# 8 models x 4 instruments x 2 respondent panels = 64 readouts.
#
# Every model runs on ONE H200 in bf16. Uniform hardware is deliberate: this
# paper's own result is that the readout path moves mass on the answer tokens
# from 0.39 to 0.99, so mixing GPU class, dtype or device count across
# candidates would confound the comparison the grid exists to make. Single-GPU
# also avoids the multi-device FP8 path that silently produced garbage here.
set -uo pipefail
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/glazkov/hf_cache
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python

MODELS=(
  "Qwen/Qwen3.8-27B:4"
  "Qwen/Qwen3.6-35B-A3B:4"
  "ibm-granite/granite-4.2-8b:8"
  "ibm-granite/granite-4.2-30b:4"
  "google/gemma-4-12B-it:8"
  "openai/gpt-oss-20b:8"
  "zai-org/GLM-4.7-Flash:8"
  "swiss-ai/Apertus-8B-Instruct-2509:8"
)
CORPORA=(ipip sd3 big5 hexaco)
PANELS=(val test2)

i=$((SLURM_ARRAY_TASK_ID - 1))
ENTRY="${MODELS[$((i / 8))]}"
M="${ENTRY%:*}"; BS="${ENTRY##*:}"
REST=$((i % 8))
C="${CORPORA[$((REST / 2))]}"
P="${PANELS[$((REST % 2))]}"
TAG=$(echo "$M" | tr '/' '_' | tr '.' 'p')

echo "=== task $SLURM_ARRAY_TASK_ID : $M / $C / $P on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" arr2026/scripts/hyp/h17_readout.py \
  --corpus "$C" --model "$M" --panel "$P" --n-panel 128 --batch-size "$BS" \
  --out "arr2026/results_euler/h17_${TAG}_${C}_${P}"
echo "DONE rc=$?"
