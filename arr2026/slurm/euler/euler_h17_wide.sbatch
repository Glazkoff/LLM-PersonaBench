#!/bin/bash
#SBATCH --job-name=arr-h17w
#SBATCH --partition=infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/h17w_%A_%a.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/h17w_%A_%a.err
# Widening pass: 3 SMALL models x 4 instruments x 2 panels = 24 readouts.
# The n=512 bootstrap resolved the proper/improper boundary but left every
# comparison inside the proper class unresolved, because all eight incumbents
# are competent instruct models and regret is normalised WITHIN the set. These
# add a weak end so the normalisation has range. The three large MoEs staged
# alongside are NOT here: they need 2-4 GPUs, and the multi-device MoE path has
# already produced garbage nondeterministically in this project, so they get
# smoke-tested before they get 24 cells.
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
  "swiss-ai/Apertus-v1.1-0.5B-Instruct:8"
  "swiss-ai/Apertus-v1.1-4B-Instruct:8"
  "google/gemma-4-E4B-it:8"
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
  --corpus "$C" --model "$M" --panel "$P" --n-panel "${NPANEL:-128}" --batch-size "$BS" \
  --out "arr2026/results_euler/h17n${NPANEL:-128}_${TAG}_${C}_${P}"
rc=$?
echo "DONE rc=$rc"
exit $rc
