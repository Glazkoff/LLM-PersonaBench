#!/bin/bash
#SBATCH --job-name=arr-a100
#SBATCH --partition=rocky
#SBATCH --qos=normal
#SBATCH -A proj_1759
#SBATCH --gres=gpu:1
#SBATCH --constraint="type_e"
#SBATCH --cpus-per-task=8
# No --mem: HSE reports RealMemory=1 on every node, so memory is not a
# scheduled resource here and any --mem request is unsatisfiable. Euler is
# the opposite -- there DefMemPerCPU gave 61GB against a 336GB checkpoint
# and OOM-killed three probes.
#SBATCH --time=08:00:00
#SBATCH --output=/home/lsavchenko/personality-arr/logs/a100_%A_%a.out
#SBATCH --error=/home/lsavchenko/personality-arr/logs/a100_%A_%a.err
# Cross-cluster replication of the belief readout.
#
# Identical script, seed and respondent panels to the Euler run; different GPU,
# different torch (2.11 vs Euler's), different transformers (5.17.0 vs 5.16.1).
# The whole H17 result set assumes a readout is a stable measurement, and this
# paper's own Limitations report the readout path moving answer-token mass from
# 0.39 to 0.99 -- so that assumption deserves a direct test.
#
# This arm runs on A100 (type_e, sm_80) against the H200 (sm_90) results, so
# it varies the GPU ARCHITECTURE rather than only the software stack.
#
# type_e was on my exclusion list as SIGILL-at-startup, but a probe showed that
# was specific to another project's venv: with this env A100 reports capability
# (8,0) and runs bf16 matmul cleanly. V100 (sm_70) is the arm that is genuinely
# unavailable -- torch 2.11+cu128 ships no Volta kernels and this cluster's
# login node cannot reach PyPI to install one.
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
  --out "arr2026/results_hse/h17a100_${TAG}_${C}_val"
rc=$?
echo "DONE rc=$rc"
exit $rc
