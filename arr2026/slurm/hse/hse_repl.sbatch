#!/bin/bash
#SBATCH --job-name=hbs-hserepl
#SBATCH --partition=rocky
#SBATCH --qos=normal
#SBATCH -A proj_1759
#SBATCH --gres=gpu:1
#SBATCH --constraint="type_h"
#SBATCH --cpus-per-task=8
# No --mem: HSE reports RealMemory=1 on every node, so memory is not a
# scheduled resource here and any --mem request is unsatisfiable. Euler is
# the opposite -- there DefMemPerCPU gave 61GB against a 336GB checkpoint
# and OOM-killed three probes.
#SBATCH --time=08:00:00
#SBATCH --output=/home/lsavchenko/personality-arr/logs/hserepl_%A_%a.out
#SBATCH --error=/home/lsavchenko/personality-arr/logs/hserepl_%A_%a.err
# Cross-cluster replication of the belief readout.
#
# Identical script, seed and respondent panels to the Euler run; different GPU,
# different torch (2.11 vs Euler's), different transformers (5.17.0 vs 5.16.1).
# The whole H17 result set assumes a readout is a stable measurement, and this
# paper's own Limitations report the readout path moving answer-token mass from
# 0.39 to 0.99 -- so that assumption deserves a direct test.
#
# constraint=type_h is the H200 nodes. type_e (a100) is excluded deliberately:
# it SIGILLs at python startup on this cluster.
set -uo pipefail
export TORCHDYNAMO_DISABLE=1        # compute nodes have GLIBC<2.34
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/lsavchenko/personality-arr/hf
export HF_HUB_OFFLINE=1             # weights are staged; never fetch from a node
cd /home/lsavchenko/personality-arr/LLM-PersonaBench
PY=/home/lsavchenko/personality-arr/arrenv/bin/python

MODELS=("swiss-ai/Apertus-v1.1-0.5B-Instruct:8" "google/gemma-4-E2B-it:8")
CORPORA=(ipip sd3)
i=$((SLURM_ARRAY_TASK_ID - 1))
E="${MODELS[$((i / 2))]}"; M="${E%:*}"; BS="${E##*:}"
C="${CORPORA[$((i % 2))]}"
TAG=$(echo "$M" | tr '/.' '__' | tr -d ' ')

echo "=== $M / $C / val on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -c "import torch,transformers;print('torch',torch.__version__,'transformers',transformers.__version__)"
"$PY" arr2026/scripts/hyp/h17_readout.py \
  --corpus "$C" --model "$M" --panel val --n-panel 512 --batch-size "$BS" \
  --out "arr2026/results_hse/h17hse_${TAG}_${C}_val"
rc=$?
echo "DONE rc=$rc"
exit $rc
