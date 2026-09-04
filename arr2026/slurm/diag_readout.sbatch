#!/bin/bash
#SBATCH --job-name=arr-diag
#SBATCH --partition=infer
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/diag_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/diag_%j.err
set -euo pipefail
export HF_HOME=$HOME/hf_cache VLLM_LOGGING_LEVEL=WARNING
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
MODEL="${1:?model}"; TP="${2:-1}"
PORT=$((8000+RANDOM%1000)); VENV=/home/glazkov/personality-twins-arr/vllmenv/bin
export PATH="$VENV:$PATH"; LOG=/home/glazkov/personality-twins-arr/logs
$VENV/python -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
  --trust-remote-code --tensor-parallel-size "$TP" --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --enforce-eager > $LOG/vllm_diag_$SLURM_JOB_ID.log 2>&1 &
P=$!; trap 'kill $P 2>/dev/null||true' EXIT
for i in $(seq 1 180); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 $P 2>/dev/null || { echo FATAL; grep -iE "error|ValueError" $LOG/vllm_diag_$SLURM_JOB_ID.log|tail -10; exit 1; }; sleep 10; done
export LOCAL_LLM_BASE_URL="http://127.0.0.1:$PORT/v1"
$VENV/python arr2026/scripts/hyp/diag_readout.py "$MODEL"
