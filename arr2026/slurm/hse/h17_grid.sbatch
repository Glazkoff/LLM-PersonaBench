#!/bin/bash
#SBATCH --job-name=arr-h17
#SBATCH --partition=rocky
#SBATCH --qos=normal
#SBATCH -A proj_1759
#SBATCH --constraint=type_h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/home/lsavchenko/personality-arr/logs/h17_%A_%a.out
#SBATCH --error=/home/lsavchenko/personality-arr/logs/h17_%A_%a.err
# 8 models x 4 instruments x 2 respondent panels = 64 readouts.
#
# Every model runs on ONE H200 in bf16. Uniform hardware is deliberate: this
# paper's own result is that the readout path moves mass on the answer tokens
# from 0.39 to 0.99, so mixing GPU classes, dtypes or device counts across
# candidates would confound the comparison the grid exists to make.
set -uo pipefail
export TORCHDYNAMO_DISABLE=1          # compute nodes have GLIBC<2.34
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/lsavchenko/personality-arr/hf
export HF_HUB_OFFLINE=1               # compute nodes have no internet
cd /home/lsavchenko/personality-arr/LLM-PersonaBench
PY=/home/lsavchenko/personality-arr/arrenv/bin/python

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

# Weights are staged on the login node while this array drains, so a task can
# start before its checkpoint has landed. Check the cache first and release the
# GPU in seconds rather than discovering it after the model load.
"$PY" - "$M" <<'PYEOF' || { echo "NOT STAGED: $M -- resubmit this task later"; exit 75; }
import sys
from transformers import AutoConfig
try:
    AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
except Exception as e:
    print(f"  {sys.argv[1]} not in local cache ({type(e).__name__})", flush=True)
    raise SystemExit(1)
print(f"  {sys.argv[1]} staged", flush=True)
PYEOF
"$PY" arr2026/scripts/hyp/h17_readout.py \
  --corpus "$C" --model "$M" --panel "$P" --n-panel 128 --batch-size "$BS" \
  --out "arr2026/results_hse/h17_${TAG}_${C}_${P}"
echo "DONE rc=$?"
