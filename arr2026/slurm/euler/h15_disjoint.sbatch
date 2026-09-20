#!/bin/bash
#SBATCH --job-name=hbs-h15
#SBATCH --partition=infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/h15_%A_%a.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/h15_%A_%a.err
# Disjoint conditioning/target replication on two instruments and two models.
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/glazkov/hf_cache
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
"$PY" -c "import xlrd" 2>/dev/null || /home/glazkov/.local/bin/uv pip install --python "$PY" --quiet xlrd

case $SLURM_ARRAY_TASK_ID in
  1) M="Qwen/Qwen2.5-7B-Instruct";   I=ipip; TAG=qwen25_7b   ; BS=8 ;;
  2) M="Qwen/Qwen2.5-7B-Instruct";   I=sd3 ; TAG=qwen25_7b   ; BS=8 ;;
  3) M="ibm-granite/granite-4.1-8b"; I=ipip; TAG=granite_8b  ; BS=8 ;;
  4) M="ibm-granite/granite-4.1-8b"; I=sd3 ; TAG=granite_8b  ; BS=8 ;;
esac

echo "=== $M / $I on $(hostname) ==="
"$PY" - <<PYEOF
from huggingface_hub import snapshot_download
print("weights at", snapshot_download("$M", max_workers=8), flush=True)
PYEOF
"$PY" arr2026/scripts/hyp/h15_disjoint_readout.py \
  --instrument "$I" --model "$M" --n-test 256 --batch-size "$BS" \
  --out "arr2026/results_euler/h15_${TAG}_${I}"
echo "DONE rc=$?"
