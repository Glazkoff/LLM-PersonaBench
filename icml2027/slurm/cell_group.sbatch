#!/bin/bash
#SBATCH --job-name=hbs-icml-cells
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
export PATH=/home/glazkov/personality-twins-arr/vllmenv/bin:$PATH
export HF_HOME=/home/glazkov/hf_cache
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false
R=/home/glazkov/personality-twins-arr/LLM-PersonaBench
cd $R
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
LOGDIR=/home/glazkov/personality-twins-arr/logs
: "${GROUPS:?}" "${MANIFEST:?}"
read MODEL SLUG CLUSTER TP DTYPE UTIL SPLIT MUT2 < <($PY -c "
import json,sys; g=json.load(open('$GROUPS'))[int('$SLURM_ARRAY_TASK_ID')]
print(g['hf_id'], g['slug'], g['cluster'], g['tp'], g['dtype'], g['gpu_util'], g.get('split', 1), g.get('mutator2') or '-')")
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
  --max-logprobs 20 --max-num-seqs 1024 --max-num-batched-tokens 65536 "${EXTRA[@]}" > "$LOGDIR/vllm_icml_${SLURM_JOB_ID}.log" 2>&1 &
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
if [ "$MUT2" != "-" ]; then
  # E8: a second mutator model served on the same GPU (the evaluated server was started at UTIL=0.55)
  MPORT=$((PORT + 700))
  $PY -m vllm.entrypoints.openai.api_server --model "$MUT2" --port "$MPORT" --host 127.0.0.1 --trust-remote-code \
    --max-model-len 16384 --gpu-memory-utilization 0.30 --enable-prefix-caching --max-num-seqs 64 \
    > "$LOGDIR/vllm_icml_mut2_${SLURM_JOB_ID}.log" 2>&1 &
  M2PID=$!; trap 'kill $VPID $M2PID 2>/dev/null || true' EXIT
  for i in $(seq 1 180); do curl -sf "http://127.0.0.1:$MPORT/v1/models" >/dev/null 2>&1 && break
    kill -0 $M2PID 2>/dev/null || { echo "FATAL: mutator2 exited"; tail -20 "$LOGDIR/vllm_icml_mut2_${SLURM_JOB_ID}.log"; exit 3; }; sleep 10; done
  export MUTATOR2_BASE_URL="http://127.0.0.1:$MPORT/v1"
  echo "mutator2 $MUT2 up on $MPORT"
fi
# one process per seed against the same server: the GPU stays busy while each process waits on the mutator
SEEDS=$($PY -c "
import json; m=json.load(open('$MANIFEST'))
print(' '.join(sorted({str(c['seed']) for c in m['cells'] if c['model_slug']=='$SLUG' and c['cluster']==$CLUSTER})))")
PIDS=()
# arms of this group split into $SPLIT chunks; one process per (seed, chunk)
CHUNKS=$($PY -c "
import json; m=json.load(open('$MANIFEST'))
arms=sorted({c['arm'] for c in m['cells'] if c['model_slug']=='$SLUG' and c['cluster']==$CLUSTER})
k=int('$SPLIT'); print(' '.join(','.join(arms[i::k]) for i in range(k)))")
for S in $SEEDS; do
  j=0
  for CH in $CHUNKS; do
    $PY -m src.optim.run_cell --manifest "$MANIFEST" --model-slug "$SLUG" --cluster "$CLUSTER" --seeds $S --arms ${CH//,/ } \
      --slurm-job-id "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" --workers 8 > "$LOGDIR/icml_run_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_s${S}_$j.log" 2>&1 &
    PIDS+=($!); j=$((j+1))
  done
done
rc=0
for P in "${PIDS[@]}"; do wait $P || rc=1; done
grep -h "completed\|failed" $LOGDIR/icml_run_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_s*.log
echo "CELLS DONE rc=$rc"
exit $rc
