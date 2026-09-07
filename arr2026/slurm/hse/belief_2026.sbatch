#!/bin/bash
#SBATCH --job-name=arr-b26
#SBATCH --partition=rocky
#SBATCH --account=proj_1759
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=/home/lsavchenko/personality-arr/logs/b26_%A_%a.out
#SBATCH --error=/home/lsavchenko/personality-arr/logs/b26_%A_%a.err

# Belief measurement for the 2026 reasoning-first models, via the HF-transformers
# path. These failed on Euler under the vLLM chat API (near-zero mass on the
# answer tokens); that path was never the one our valid runs used, so this tests
# whether the failure was transport-specific rather than model-specific.
set -euo pipefail
export TORCHDYNAMO_DISABLE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/lsavchenko/hf_cache
# compressed-tensors (+loguru) for the FP8 checkpoint; the stock env lacks them.
export PYTHONPATH=/home/lsavchenko/personality-arr/pylibs${PYTHONPATH:+:$PYTHONPATH}
cd /home/lsavchenko/personality-arr/LLM-PersonaBench

case $SLURM_ARRAY_TASK_ID in
  1) MODEL="Qwen/Qwen3.5-9B";            TAG=qwen35_9b;   BS=8 ;;
  2) MODEL="zai-org/GLM-4.7-Flash";      TAG=glm47_flash; BS=4 ;;
  3) MODEL="Qwen/Qwen3.6-27B-FP8";       TAG=qwen36_27b;  BS=4 ;;
esac

# HSE already ships a modern stack; no venv build needed (and the compute nodes
# time out against download.pytorch.org, so building one is not an option).
# NB the env names are misleading: ..._2024 ships transformers 5.2.0 (which knows
# qwen3_5 / glm4_moe_lite), while ..._2025 is stuck on 4.55.2 and cannot even build
# the configs. Verified on the login node before submitting.
PY=/opt/software/python/envs/google_colab_gpu_2024/bin/python3
[ -x "$PY" ] || { echo "FATAL: $PY missing"; exit 1; }
"$PY" -c "
from transformers import AutoConfig; AutoConfig.from_pretrained('$MODEL')" \
  || { echo 'FATAL: this transformers cannot build the config for $MODEL'; exit 1; }
"$PY" -c "import torch,transformers,pandas; print('torch',torch.__version__,'tf',transformers.__version__)"

echo "=== $MODEL on $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "--- fetching weights (compute nodes are online) ---"
"$PY" - <<PYEOF
from huggingface_hub import snapshot_download
p = snapshot_download("$MODEL", max_workers=8)
print("weights at", p, flush=True)
PYEOF

"$PY" arr2026/scripts/hyp/h4_variance_ladder_hf.py \
  --model "$MODEL" --clusters 0 1 2 3 --n-personas 40 \
  --batch-size "$BS" --out "arr2026/results/h4_$TAG"
echo "DONE rc=$?"
