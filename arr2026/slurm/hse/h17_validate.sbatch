#!/bin/bash
#SBATCH --job-name=arr-h17val
#SBATCH --partition=rocky
#SBATCH --qos=normal
#SBATCH -A proj_1759
#SBATCH --constraint=type_h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/home/lsavchenko/personality-arr/logs/h17val_%j.out
#SBATCH --error=/home/lsavchenko/personality-arr/logs/h17val_%j.err
# Smallest cell of the grid: one small model, the smallest instrument, dumping
# the first prompt. Validates loader, K handling, orientation, panels, answer
# tokens and mass BEFORE 64 jobs are queued behind a 400GB download.
set -uo pipefail
export TORCHDYNAMO_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/lsavchenko/personality-arr/hf
export HF_HUB_OFFLINE=1
cd /home/lsavchenko/personality-arr/LLM-PersonaBench
PY=/home/lsavchenko/personality-arr/arrenv/bin/python
echo "host=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for C in sd3 hexaco; do
  echo "########## $C ##########"
  "$PY" arr2026/scripts/hyp/h17_readout.py --corpus "$C" \
    --model ibm-granite/granite-4.2-8b --panel val --n-panel 8 \
    --batch-size 8 --debug-dump 2 \
    --out "/tmp/h17_validate_${C}" 2>&1 | grep -vE "it/s\]|^Loading|^Fetching"
done
echo "VALIDATE DONE"
