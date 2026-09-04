#!/bin/bash
#SBATCH --job-name=arr-h4
#SBATCH --partition=infer
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/h4_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/h4_%j.err

# usage: sbatch --gres=gpu:N arr2026/slurm/h4_model.sbatch <model_id> <tp> <tag>
set -euo pipefail
export HF_HOME=$HOME/hf_cache
export VLLM_LOGGING_LEVEL=WARNING
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench

MODEL="${1:?model id required}"
TP="${2:-1}"
TAG="${3:-run}"
PORT=$((8000 + RANDOM % 1000))
VENV=/home/glazkov/personality-twins-arr/vllmenv/bin
export PATH="$VENV:$PATH"
LOGDIR=/home/glazkov/personality-twins-arr/logs

echo "=== $MODEL  TP=$TP  port=$PORT ==="
$VENV/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port $PORT --trust-remote-code \
  --tensor-parallel-size "$TP" \
  --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --enforce-eager > $LOGDIR/vllm_$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

UP=0
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "server up after $((i*10))s"; UP=1; break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "FATAL: vLLM exited"; grep -E "Error|error|ValueError" $LOGDIR/vllm_$SLURM_JOB_ID.log | tail -20; exit 1
  fi
  sleep 10
done
[ "$UP" = "1" ] || { echo "FATAL: server never came up"; tail -40 $LOGDIR/vllm_$SLURM_JOB_ID.log; exit 1; }

export LOCAL_LLM_BASE_URL="http://127.0.0.1:$PORT/v1"
/home/glazkov/.local/bin/uv run --with pandas --with numpy --with openai \
  --with langchain-openai --with langchain-core --with python-dotenv \
  python arr2026/scripts/hyp/h4_variance_ladder.py \
    --model "$MODEL" --clusters 0 1 2 3 --n-personas 40 --workers 48 \
    --out "arr2026/results/h4_$TAG"
echo "H4 DONE rc=$?"
