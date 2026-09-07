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
esac

PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
[ -x "$PY" ] || { echo "FATAL: $PY missing"; exit 1; }
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
