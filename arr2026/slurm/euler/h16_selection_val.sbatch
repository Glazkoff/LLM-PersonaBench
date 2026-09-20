#!/bin/bash
#SBATCH --job-name=hbs-h16val
#SBATCH --partition=infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/h16val_%A_%a.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/h16val_%A_%a.err
# H16 step 1: score all eleven belief models on a VALIDATION panel of 128
# respondents that is disjoint from the training block and from App. B's 256.
# Two selection rules are then applied to these same forecasts; only the models
# they pick get run on the fresh test panel.
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/glazkov/hf_cache
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python

case $SLURM_ARRAY_TASK_ID in
   1) M="Qwen/Qwen2.5-7B-Instruct";              TAG=qwen25_7b   ; BS=8 ;;
   2) M="Qwen/Qwen2.5-32B-Instruct";             TAG=qwen25_32b  ; BS=8 ;;
   3) M="Qwen/Qwen2.5-72B-Instruct";             TAG=qwen25_72b  ; BS=4 ;;
   4) M="mistralai/Mistral-Small-24B-Instruct-2501"; TAG=mistral_24b; BS=8 ;;
   5) M="Qwen/QwQ-32B-Preview";                  TAG=qwq_32b     ; BS=8 ;;
   6) M="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"; TAG=qwen3_235b ; BS=2 ;;
   7) M="Qwen/Qwen3.5-9B";                       TAG=qwen35_9b   ; BS=8 ;;
   8) M="zai-org/GLM-4.7-Flash";                 TAG=glm47_flash ; BS=8 ;;
   9) M="openai/gpt-oss-20b";                    TAG=gptoss_20b  ; BS=8 ;;
  10) M="ibm-granite/granite-4.1-8b";            TAG=granite_8b  ; BS=8 ;;
  11) M="swiss-ai/Apertus-8B-Instruct-2509";     TAG=apertus_8b  ; BS=8 ;;
esac

echo "=== $M on $(hostname) : IPIP validation panel ==="
"$PY" - <<PYEOF
from huggingface_hub import snapshot_download
print("weights at", snapshot_download("$M", max_workers=8), flush=True)
PYEOF
"$PY" arr2026/scripts/hyp/h15_disjoint_readout.py \
  --instrument ipip --model "$M" --panel val --n-panel 128 --batch-size "$BS" \
  --out "arr2026/results_euler/h16_val_${TAG}_ipip"
echo "DONE rc=$?"
