#!/bin/bash
#SBATCH --job-name=hbs-icml-transfer
#SBATCH --partition=infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/icml_transfer_%A_%a.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/icml_transfer_%A_%a.err
# One array task per Tier-1 model (E6/H4/H6). Serves the model once and runs, idempotently:
#   1. IPIP-NEO-300 unseen-item transfer + m* for every main-fitness cell of this model
#   2. cross-cluster transfer (same model, other clusters) for arms ga/gepa/bestofb/random, seed 1
#   3. cross-model transfer with THIS model as the target, for other models' cells, same arms, seed 1
# Submitted by icml2027/slurm/stage.sbatch with --array=0-(n-1)%3.
set -uo pipefail
export PATH=/home/glazkov/personality-twins-arr/vllmenv/bin:$PATH
export HF_HOME=/home/glazkov/hf_cache VLLM_LOGGING_LEVEL=WARNING TOKENIZERS_PARALLELISM=false
R=/home/glazkov/personality-twins-arr/LLM-PersonaBench; cd $R
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python; LOGDIR=/home/glazkov/personality-twins-arr/logs
read MODEL SLUG PREFILL_B64 < <($PY -c "
import yaml,base64; d=yaml.safe_load(open('icml2027/configs/models.yaml')); df=d['defaults']
ms=[m for m in d['models'] if m.get('tier')==1]; m=ms[int('$SLURM_ARRAY_TASK_ID')]
print(m['hf_id'], m['slug'], base64.b64encode(m.get('prefill',df['prefill']).encode()).decode())")
PREFILL=$(echo "$PREFILL_B64" | base64 -d)
PORT=$((8900 + SLURM_ARRAY_TASK_ID))
echo "=== transfer: $MODEL ($SLUG) ==="
$PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT --host 127.0.0.1 --trust-remote-code \
  --max-model-len 4096 --gpu-memory-utilization 0.90 --enable-prefix-caching --max-logprobs 20 \
  --max-num-seqs 1024 --max-num-batched-tokens 65536 > "$LOGDIR/vllm_transfer_${SLURM_JOB_ID}.log" 2>&1 &
VPID=$!; trap 'kill $VPID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 $VPID 2>/dev/null || { echo "FATAL vLLM exited"; tail -30 "$LOGDIR/vllm_transfer_${SLURM_JOB_ID}.log"; exit 3; }; sleep 10; done
export LOCAL_LLM_BASE_URL="http://127.0.0.1:$PORT/v1"
rc=0
$PY -m src.optim.transfer ipip300 --model "$MODEL" --slug "$SLUG" --prefill "$PREFILL" --cells-glob "*__${SLUG}__*" || rc=1
$PY -m src.optim.transfer crosscluster --model "$MODEL" --slug "$SLUG" --prefill "$PREFILL" --cells-glob "*__${SLUG}__*" \
  --arms ga gepa bestofb random --seeds 1 || rc=1
$PY -m src.optim.transfer crossmodel --model "$MODEL" --slug "$SLUG" --prefill "$PREFILL" --cells-glob "*" \
  --arms ga gepa bestofb random --seeds 1 || rc=1
echo "TRANSFER DONE rc=$rc"; exit $rc
