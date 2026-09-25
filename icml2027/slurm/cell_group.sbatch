#!/bin/bash
#SBATCH --job-name=icml-cells
#SBATCH --partition=infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/icml_cells_%A_%a.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/icml_cells_%A_%a.err
# One array task = one (model, cluster): serve the model once, run every (arm, seed) cell
# listed for that cluster in the manifest, sequentially, against the same server.
# Submitted by icml2027/scripts/queue.py, which sets:
#   --export=ALL,MODEL=<hf_id>,SLUG=<slug>,TP=<n>,DTYPE=<dtype>,GPU_UTIL=<f>,MANIFEST=<path>
#   --array=0-3%<concurrency>   (SLURM_ARRAY_TASK_ID = cluster)
set -uo pipefail
export HF_HOME=/home/glazkov/hf_cache
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
VENV=/home/glazkov/personality-twins-arr/vllmenv/bin
PY=$VENV/python
LOGDIR=/home/glazkov/personality-twins-arr/logs
CLUSTER=$SLURM_ARRAY_TASK_ID
PORT=$((8000 + RANDOM % 1000))
: "${MODEL:?}" "${SLUG:?}" "${MANIFEST:?}"
TP="${TP:-1}"; GPU_UTIL="${GPU_UTIL:-0.90}"

echo "=== $MODEL ($SLUG) cluster=$CLUSTER TP=$TP port=$PORT node=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

EXTRA=()
[ "${DTYPE:-bf16}" = "fp8" ] && EXTRA+=(--quantization fp8)
$PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port "$PORT" --trust-remote-code \
  --tensor-parallel-size "$TP" --max-model-len 8192 \
  --gpu-memory-utilization "$GPU_UTIL" --enable-prefix-caching \
  --max-logprobs 20 "${EXTRA[@]}" > "$LOGDIR/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

UP=0
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "server up after $((i*10))s"; UP=1; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "FATAL: vLLM exited"; grep -iE "error" "$LOGDIR/vllm_${SLURM_JOB_ID}.log" | tail -20; exit 3; }
  sleep 10
done
[ "$UP" = 1 ] || { echo "FATAL: server never came up"; tail -40 "$LOGDIR/vllm_${SLURM_JOB_ID}.log"; exit 3; }

export LOCAL_LLM_BASE_URL="http://127.0.0.1:$PORT/v1"
# run_cell iterates the manifest's cells for (SLUG, CLUSTER); each writes status.json itself.
$PY -m src.optim.run_cell --manifest "$MANIFEST" --model-slug "$SLUG" --cluster "$CLUSTER" \
  --slurm-job-id "${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" --workers 48
rc=$?
echo "CELLS DONE rc=$rc"
exit $rc
