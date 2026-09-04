#!/bin/bash
#SBATCH --job-name=arr-diag
#SBATCH --partition=infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/diag_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/diag_%j.err
set -euo pipefail
export HF_HOME=$HOME/hf_cache VLLM_LOGGING_LEVEL=WARNING
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
MODEL="${1:-Qwen/Qwen3-8B}"; PORT=$((8000+RANDOM%1000))
VENV=/home/glazkov/personality-twins-arr/vllmenv/bin; export PATH="$VENV:$PATH"
LOGDIR=/home/glazkov/personality-twins-arr/logs
$VENV/python -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
  --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --enforce-eager > $LOGDIR/vllm_$SLURM_JOB_ID.log 2>&1 &
P=$!; trap 'kill $P 2>/dev/null||true' EXIT
for i in $(seq 1 180); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break; kill -0 $P 2>/dev/null || { echo FATAL; tail -20 $LOGDIR/vllm_$SLURM_JOB_ID.log; exit 1; }; sleep 10; done
export LOCAL_LLM_BASE_URL="http://127.0.0.1:$PORT/v1"
/home/glazkov/.local/bin/uv run --with openai --with langchain-openai --with langchain-core \
  --with python-dotenv python arr2026/scripts/hyp/diag_logprobs.py "$MODEL"
