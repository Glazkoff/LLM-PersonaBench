#!/bin/bash
#SBATCH --job-name=arr-h4hse
#SBATCH --partition=rocky
#SBATCH --account=proj_1759
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --constraint="type_a|type_b|type_c"
#SBATCH --output=/home/lsavchenko/personality-arr/logs/h4hse_%A_%a.out
#SBATCH --error=/home/lsavchenko/personality-arr/logs/h4hse_%A_%a.err

# Replicates the belief-level variance measurement across the models cached on
# HSE, which is what turns H4 from a single-model observation into a claim.
set -euo pipefail
export TORCHDYNAMO_DISABLE=1
export HF_HOME=/home/lsavchenko/imak-exp/.shared-cache/hf
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING
cd /home/lsavchenko/personality-arr/LLM-PersonaBench

case $SLURM_ARRAY_TASK_ID in
  1) MODEL="Qwen/Qwen2.5-7B-Instruct";                TP=1; TAG=qwen25_7b ;;
  2) MODEL="Qwen/Qwen2.5-32B-Instruct";               TP=2; TAG=qwen25_32b ;;
  3) MODEL="Qwen/Qwen2.5-72B-Instruct";               TP=4; TAG=qwen25_72b ;;
  4) MODEL="mistralai/Mistral-Small-24B-Instruct-2501"; TP=2; TAG=mistral_24b ;;
  5) MODEL="Qwen/QwQ-32B-Preview";                    TP=2; TAG=qwq_32b ;;
esac

PORT=$((8000 + RANDOM % 1000))
VENV=/home/lsavchenko/personality-arr/vllmenv/bin
export PATH="$VENV:$PATH"
LOGDIR=/home/lsavchenko/personality-arr/logs

echo "=== $MODEL TP=$TP port=$PORT on $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

$VENV/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port $PORT --trust-remote-code \
  --tensor-parallel-size "$TP" --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --enable-prefix-caching --enforce-eager \
  > $LOGDIR/vllm_hse_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1 &
P=$!; trap 'kill $P 2>/dev/null || true' EXIT

UP=0
for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "server up after $((i*10))s"; UP=1; break; }
  kill -0 $P 2>/dev/null || { echo "FATAL: vLLM exited"; grep -iE "error|ValueError|not supported" $LOGDIR/vllm_hse_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log | tail -15; exit 1; }
  sleep 10
done
[ "$UP" = "1" ] || { echo "FATAL: server never came up"; tail -30 $LOGDIR/vllm_hse_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log; exit 1; }

export LOCAL_LLM_BASE_URL="http://127.0.0.1:$PORT/v1"
$VENV/python arr2026/scripts/hyp/h4_variance_ladder.py \
  --model "$MODEL" --clusters 0 1 2 3 --n-personas 40 --workers 48 \
  --out "arr2026/results/h4_$TAG"
echo "H4 HSE DONE rc=$?"
