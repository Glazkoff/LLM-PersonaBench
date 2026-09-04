#!/bin/bash
#SBATCH --job-name=arr-h4hf
#SBATCH --partition=rocky
#SBATCH --account=proj_1759
#SBATCH --qos=normal
# NOTE: no --mem on cHARISMa; memory is not a schedulable TRES here.
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=/home/lsavchenko/personality-arr/logs/h4hf_%A_%a.out
#SBATCH --error=/home/lsavchenko/personality-arr/logs/h4hf_%A_%a.err

# Belief-level variance measurement across the models cached on HSE. This is
# what turns the single-model result into a claim about persona simulators.
set -euo pipefail
export TORCHDYNAMO_DISABLE=1
export HF_HOME=/home/lsavchenko/imak-exp/.shared-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
module load Python/PyTorch_GPU_v2.4 2>/dev/null || true
cd /home/lsavchenko/personality-arr/LLM-PersonaBench

case $SLURM_ARRAY_TASK_ID in
  1) MODEL="Qwen/Qwen2.5-7B-Instruct";                  TAG=qwen25_7b;   BS=16 ;;
  2) MODEL="Qwen/Qwen2.5-32B-Instruct";                 TAG=qwen25_32b;  BS=8  ;;
  3) MODEL="Qwen/Qwen2.5-72B-Instruct";                 TAG=qwen25_72b;  BS=4  ;;
  4) MODEL="mistralai/Mistral-Small-24B-Instruct-2501"; TAG=mistral_24b; BS=8  ;;
  5) MODEL="Qwen/QwQ-32B-Preview";                      TAG=qwq_32b;     BS=8  ;;
esac

echo "=== $MODEL on $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python3 arr2026/scripts/hyp/h4_variance_ladder_hf.py \
  --model "$MODEL" --clusters 0 1 2 3 --n-personas 40 \
  --batch-size "$BS" --out "arr2026/results/h4_$TAG"
echo "H4-HF DONE rc=$?"
