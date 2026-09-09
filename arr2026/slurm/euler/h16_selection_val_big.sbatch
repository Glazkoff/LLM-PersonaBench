#!/bin/bash
#SBATCH --job-name=arr-h16big
#SBATCH --partition=infer
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/h16big_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/h16big_%j.err
# The two candidates that failed in array 1201, re-run with enough GPUs.
#   Qwen2.5-72B  bf16 ~145GB -> 2x H200; task 3 died on a transient HF xet CDN error
#   Qwen3-235B   FP8  ~235GB -> 4x H200; task 6 died in the FP8 quantizer, and
#                                        would not have fit on the one GPU anyway
# M, TAG and BS come from --export.
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/glazkov/hf_cache
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python

echo "=== $M on $(hostname), $(nvidia-smi -L | wc -l) GPUs visible ==="
for try in 1 2 3; do
  "$PY" - <<PYEOF && break || { echo "download attempt $try failed; retrying"; sleep 60; }
from huggingface_hub import snapshot_download
print("weights at", snapshot_download("$M", max_workers=8), flush=True)
PYEOF
done
"$PY" arr2026/scripts/hyp/h15_disjoint_readout.py \
  --instrument ipip --model "$M" --panel val --n-panel 128 --batch-size "$BS" \
  --out "arr2026/results_euler/h16_val_${TAG}_ipip"
echo "DONE rc=$?"
