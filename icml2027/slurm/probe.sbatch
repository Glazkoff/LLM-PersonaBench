#!/bin/bash
#SBATCH --job-name=icml-probe
#SBATCH --partition=infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/icml_probe_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/icml_probe_%j.err
# Readout guard for every model listed in $MODELS ("hf_id|slug|prefill|dtype" per line, one GPU, sequential).
set -uo pipefail
export PATH=/home/glazkov/personality-twins-arr/vllmenv/bin:$PATH
export HF_HOME=/home/glazkov/hf_cache VLLM_LOGGING_LEVEL=WARNING TOKENIZERS_PARALLELISM=false
R=/home/glazkov/personality-twins-arr/LLM-PersonaBench; cd $R
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python; LOGDIR=/home/glazkov/personality-twins-arr/logs
mkdir -p icml2027/results/aggregates/probe
PORT=$((8700 + RANDOM % 80))
while IFS='|' read -r MODEL SLUG PREFILL DTYPE; do
  [ -z "$MODEL" ] && continue
  EXTRA=(); [ "$DTYPE" = "fp8" ] && EXTRA+=(--quantization fp8)
  $PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT --host 127.0.0.1 --trust-remote-code \
    --max-model-len 4096 --gpu-memory-utilization 0.90 --enable-prefix-caching --max-logprobs 20 --max-num-seqs 512 \
    "${EXTRA[@]}" > "$LOGDIR/vllm_probe_${SLURM_JOB_ID}_${SLUG}.log" 2>&1 &
  VPID=$!; UP=0
  for i in $(seq 1 180); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { UP=1; break; }; kill -0 $VPID 2>/dev/null || break; sleep 10; done
  if [ "$UP" = 1 ]; then
    LOCAL_LLM_BASE_URL="http://127.0.0.1:$PORT/v1" $PY -m src.optim.probe --model "$MODEL" --prefill "$PREFILL" \
      --out "icml2027/results/aggregates/probe/$SLUG.json"; echo "PROBE $SLUG rc=$?"
  else
    echo "PROBE $SLUG server_failed"; tail -15 "$LOGDIR/vllm_probe_${SLURM_JOB_ID}_${SLUG}.log"
  fi
  kill $VPID 2>/dev/null; wait $VPID 2>/dev/null; sleep 5
done <<< "$MODELS"
