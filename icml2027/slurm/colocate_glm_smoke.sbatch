#!/bin/bash
#SBATCH --job-name=hbs-icml-smoke
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/icml_smoke_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/icml_smoke_%j.err
# Engineering-only load test on non-evaluation respondents. No cell results are written.
set -euo pipefail
R=/home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
LOGDIR=/home/glazkov/personality-twins-arr/logs
export PATH=/home/glazkov/personality-twins-arr/vllmenv/bin:$PATH
export HF_HOME=/home/glazkov/hf_cache
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=WARNING
cd "$R"
MODEL=zai-org/GLM-4.7-Flash
MUTATOR=Qwen/Qwen3.8-27B
PORT=$((20000 + SLURM_JOB_ID % 10000))
MPORT=$((30000 + SLURM_JOB_ID % 10000))
TMP=$(mktemp -d "/tmp/hbs-glm-smoke-${SLURM_JOB_ID}.XXXX")
PIDS=()
cleanup() {
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
  for p in "${PIDS[@]}"; do wait "$p" 2>/dev/null || true; done
  rm -rf "$TMP"
}
trap cleanup EXIT
echo "job=$SLURM_JOB_ID model=$MODEL mutator=$MUTATOR target_util=0.45 mutator_util=0.42 max_seqs=256 batch_tokens=16384"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

"$PY" -m vllm.entrypoints.openai.api_server --model "$MUTATOR" --port "$MPORT" --host 127.0.0.1 \
  --trust-remote-code --max-model-len 16384 --gpu-memory-utilization 0.42 \
  --enable-prefix-caching --max-num-seqs 64 > "$LOGDIR/vllm_smoke_mut_${SLURM_JOB_ID}.log" 2>&1 &
PIDS+=($!)
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$MPORT/v1/models" >/dev/null 2>&1 && break
  kill -0 "${PIDS[0]}" 2>/dev/null || { tail -30 "$LOGDIR/vllm_smoke_mut_${SLURM_JOB_ID}.log"; exit 3; }
  sleep 5
done
curl -sf "http://127.0.0.1:$MPORT/v1/models" >/dev/null

"$PY" -m vllm.entrypoints.openai.api_server --model "$MODEL" --port "$PORT" --host 127.0.0.1 \
  --trust-remote-code --max-model-len 4096 --gpu-memory-utilization 0.45 --enable-prefix-caching \
  --max-logprobs 20 --max-num-seqs 256 --max-num-batched-tokens 16384 \
  > "$LOGDIR/vllm_smoke_glm_${SLURM_JOB_ID}.log" 2>&1 &
PIDS+=($!)
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 "${PIDS[1]}" 2>/dev/null || { tail -30 "$LOGDIR/vllm_smoke_glm_${SLURM_JOB_ID}.log"; exit 3; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null

"$PY" icml2027/scripts/smoke_mutator.py --base-url "http://127.0.0.1:$MPORT/v1" \
  --model "$MUTATOR" --requests 120 > "$TMP/mutator.log" 2>&1 &
MLOAD=$!
PIDS+=($MLOAD)
for round in 1 2 3 4 5; do
  probes=()
  for i in 1 2 3 4 5 6; do
    LOCAL_LLM_BASE_URL="http://127.0.0.1:$PORT/v1" "$PY" -m src.optim.probe \
      --model "$MODEL" --n 200 --out "$TMP/probe_${round}_${i}.json" \
      > "$TMP/probe_${round}_${i}.log" 2>&1 &
    probes+=($!)
  done
  for p in "${probes[@]}"; do wait "$p" || { echo "probe failed in round $round"; exit 4; }; done
  echo "round=$round probes=6 passed"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
done
wait "$MLOAD"
tail -1 "$TMP/mutator.log"
if grep -Eiq 'memory allocation failed with OOM|CUDA out of memory|OutOfMemoryError' \
   "$LOGDIR/vllm_smoke_glm_${SLURM_JOB_ID}.log" "$LOGDIR/vllm_smoke_mut_${SLURM_JOB_ID}.log"; then
  echo "FAIL: vLLM reported an out-of-memory condition"
  exit 5
fi
echo "PASS: colocated GLM load smoke completed without OOM"
