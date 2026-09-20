#!/bin/bash
#SBATCH --job-name=hbs-h24h
#SBATCH --partition=rocky
#SBATCH --qos=normal
#SBATCH -A proj_1759
#SBATCH --gres=gpu:1
#SBATCH --constraint="type_f|type_h"
# type_f = H100 (cn-047,048), type_h = H200 (cn-049..051). NOT type_a|b|c (V100,
# sm_70): arrenv's torch is a cu128 build with no sm_70 kernels, and the first
# array attempt died on every V100 with
#   torch.AcceleratorError: CUDA error: no kernel image is available
# after loading the corpus correctly. type_e (a100) stays excluded: it SIGILLs
# at python startup on this cluster.
# No --mem: cHARISMa reports RealMemory=1, so memory is not a schedulable TRES.
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --array=1-4%2
#SBATCH --output=/home/lsavchenko/personality-arr/logs/h24h_%A_%a.out
#SBATCH --error=/home/lsavchenko/personality-arr/logs/h24h_%A_%a.err
# H24 on cHARISMa: the same IPIP-NEO-300 unseen-item-family readout Euler runs,
# on the two models staged here and at two history budgets. Small models extend
# the transfer result down the scale ladder, which Euler's set does not cover.
#
# Cross-cluster by design: a different scheduler, a different torch/transformers
# pair (2.11.0/5.17.0 against Euler's 2.13.0/5.16.x) and different physical GPUs,
# so a matching result is evidence the readout is a property of the model rather
# than of the machine.
set -uo pipefail
export TORCHDYNAMO_DISABLE=1        # compute nodes have GLIBC<2.34
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/lsavchenko/personality-arr/hf
export HF_HUB_OFFLINE=1             # weights are staged; nodes have no internet
cd /home/lsavchenko/personality-arr/LLM-PersonaBench
PY=/home/lsavchenko/personality-arr/arrenv/bin/python

case $SLURM_ARRAY_TASK_ID in
  1) M="google/gemma-4-E2B-it";                MODE=scores;  OBS=0;  TAG=gemma4_e2b ;;
  2) M="swiss-ai/Apertus-v1.1-0.5B-Instruct";  MODE=scores;  OBS=0;  TAG=apertus_05b ;;
  3) M="google/gemma-4-E2B-it";                MODE=history; OBS=20; TAG=gemma4_e2b ;;
  4) M="swiss-ai/Apertus-v1.1-0.5B-Instruct";  MODE=history; OBS=20; TAG=apertus_05b ;;
  *) echo "no such task"; exit 2 ;;
esac
if [ "$MODE" = scores ]; then OUT="arr2026/results_hse/h24_${TAG}_scores"
else OUT="arr2026/results_hse/h24_${TAG}_hist${OBS}"; fi

echo "=== task $SLURM_ARRAY_TASK_ID: $M / $MODE obs=$OBS on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -c "import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__)"
"$PY" arr2026/scripts/hyp/h24_ipip300_readout.py \
  --model "$M" --persona "$MODE" --n-observed "$OBS" \
  --n-scored 150 --batch-size 8 --out "$OUT"
rc=$?
echo "DONE rc=$rc"
exit $rc
