#!/bin/bash
#SBATCH --job-name=arr-h4-giga
#SBATCH --partition=infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/h4giga_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/h4giga_%j.err

set -euo pipefail
export HF_HOME=$HOME/hf_cache
export VLLM_LOGGING_LEVEL=WARNING
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench

MODEL="ai-sage/GigaChat3-10B-A1.8B"
PORT=$((8000 + RANDOM % 1000))
VENV=/home/glazkov/personality-twins-arr/vllmenv/bin
LOGDIR=/home/glazkov/personality-twins-arr/logs

echo "=== launching vLLM on port $PORT ==="
$VENV/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port $PORT --trust-remote-code \
  --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching > $LOGDIR/vllm_$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

echo "=== waiting for server ==="
UP=0
for i in $(seq 1 150); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "server up after $((i*10))s"; UP=1; break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "FATAL: vLLM exited"; tail -60 $LOGDIR/vllm_$SLURM_JOB_ID.log; exit 1
  fi
  sleep 10
done
if [ "$UP" != "1" ]; then
  echo "FATAL: server never came up"; tail -60 $LOGDIR/vllm_$SLURM_JOB_ID.log; exit 1
fi

export LOCAL_LLM_BASE_URL="http://127.0.0.1:$PORT/v1"
/home/glazkov/.local/bin/uv run --with pandas --with numpy --with openai \
  --with langchain-openai --with langchain-core --with python-dotenv \
  python arr2026/scripts/hyp/h4_variance_ladder.py \
    --model "$MODEL" --clusters 0 1 2 3 --n-personas 40 --workers 48 \
    --out arr2026/results/h4_gigachat
echo "H4 DONE rc=$?"
