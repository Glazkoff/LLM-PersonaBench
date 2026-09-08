#!/bin/bash
#SBATCH --job-name=arr-fam
#SBATCH --partition=infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/fam_%A_%a.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/fam_%A_%a.err
# Belief measurement for families not yet in the table. Everything -- including
# the weight download -- runs inside the Slurm allocation.
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/glazkov/hf_cache
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench

case $SLURM_ARRAY_TASK_ID in
  1) MODEL="openai/gpt-oss-20b";                       TAG=gptoss_20b;  BS=4 ;;
  2) MODEL="deepseek-ai/DeepSeek-V4-Flash";            TAG=dsv4_flash;  BS=4 ;;
  3) MODEL="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16";    TAG=nemotron_4b; BS=8 ;;
  # 3 needs mamba-ssm (hybrid Mamba); replaced by two plain-transformer families.
  4) MODEL="ibm-granite/granite-4.1-8b";               TAG=granite_8b;  BS=8 ;;
  5) MODEL="swiss-ai/Apertus-8B-Instruct-2509";        TAG=apertus_8b;  BS=8 ;;
  # FP8 gave coverage 0.0 (all-NaN logits) on HSE under torch 2.5.1; retry here,
  # where the H200s support FP8 natively and torch is 2.13.
  6) MODEL="Qwen/Qwen3.6-27B-FP8";                     TAG=qwen36_27b;  BS=4 ;;
  # 2 (DeepSeek-V4-Flash) is 291B / ~290GB and stalled out its walltime in the
  # download; this Lite checkpoint represents the same family at 15.7B.
  7) MODEL="deepseek-ai/DeepSeek-V2-Lite-Chat";        TAG=dsv2_lite;   BS=8 ;;
esac

PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
[ -x "$PY" ] || { echo "FATAL: $PY missing"; exit 1; }
# FP8 checkpoints need compressed-tensors; add it inside the allocation if absent.
"$PY" -c "import compressed_tensors" 2>/dev/null \
  || /home/glazkov/.local/bin/uv pip install --python "$PY" --quiet compressed-tensors
"$PY" -c "import torch,transformers,pandas,accelerate; \
  print('torch',torch.__version__,'tf',transformers.__version__)"

# Fail in seconds on an unsupported architecture rather than after the download.
"$PY" -c "
from transformers import AutoConfig
print('config ok:', type(AutoConfig.from_pretrained('$MODEL')).__name__)"

echo "=== $MODEL on $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$PY" - <<PYEOF
from huggingface_hub import snapshot_download
print("weights at", snapshot_download("$MODEL", max_workers=8), flush=True)
PYEOF

"$PY" arr2026/scripts/hyp/h4_variance_ladder_hf.py \
  --model "$MODEL" --clusters 0 1 2 3 --n-personas 40 \
  --batch-size "$BS" --out "arr2026/results_euler/h4_$TAG"
echo "DONE rc=$?"
