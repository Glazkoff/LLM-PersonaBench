#!/bin/bash
#SBATCH --job-name=icml-cells
#SBATCH --partition=infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/icml_cells_%A_%a.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/icml_cells_%A_%a.err
# One array task = one (model, cluster) group from $GROUPS: serve the model once and run every cell of
# that group in the manifest. Submitted by icml2027/scripts/queue.py with GROUPS and MANIFEST exported.
set -uo pipefail
export HF_HOME=/home/glazkov/hf_cache
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false
R=/home/glazkov/personality-twins-arr/LLM-PersonaBench
cd $R
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
LOGDIR=/home/glazkov/personality-twins-arr/logs
: "${GROUPS:?}" "${MANIFEST:?}"
read MODEL SLUG CLUSTER TP DTYPE UTIL < <($PY -c "
import json,sys; g=json.load(open('$GROUPS'))[int('$SLURM_ARRAY_TASK_ID')]
print(g['hf_id'], g['slug'], g['cluster'], g['tp'], g['dtype'], g['gpu_util'])")
PORT=$((8100 + SLURM_ARRAY_TASK_ID % 600))
export GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "=== $MODEL ($SLUG) cluster=$CLUSTER TP=$TP port=$PORT node=$(hostname) gpu=$GPU_NAME ==="

for i in $(seq 1 360); do [ -f icml2027/queue/mutator.json ] && break; sleep 10; done
[ -f icml2027/queue/mutator.json ] || { echo "FATAL: no mutator service"; exit 4; }
export MUTATOR_BASE_URL=$($PY -c "import json;print(json.load(open('icml2027/queue/mutator.json'))['url'])")
curl -sf "$MUTATOR_BASE_URL/models" >/dev/null || { echo "FATAL: mutator not answering at $MUTATOR_BASE_URL"; exit 4; }

EXTRA=()
[ "$DTYPE" = "fp8" ] && EXTRA+=(--quantization fp8)
$PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --port "$PORT" --host 127.0.0.1 --trust-remote-code \
  --tensor-parallel-size "$TP" --max-model-len 4096 --gpu-memory-utilization "$UTIL" --enable-prefix-caching \
  --max-logprobs 20 --max-num-seqs 512 "${EXTRA[@]}" > "$LOGDIR/vllm_icml_${SLURM_JOB_ID}.log" 2>&1 &
VPID=$!
trap 'kill $VPID 2>/dev/null || true' EXIT
UP=0
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "server up after $((i*10))s"; UP=1; break; }
  kill -0 $VPID 2>/dev/null || { echo "FATAL: vLLM exited"; tail -30 "$LOGDIR/vllm_icml_${SLURM_JOB_ID}.log"; exit 3; }
  sleep 10
done
[ "$UP" = 1 ] || { echo "FATAL: server never came up"; exit 3; }
export LOCAL_LLM_BASE_URL="http://127.0.0.1:$PORT/v1"
$PY -m src.optim.run_cell --manifest "$MANIFEST" --model-slug "$SLUG" --cluster "$CLUSTER" \
  --slurm-job-id "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" --workers 16
rc=$?
echo "CELLS DONE rc=$rc"
exit $rc
