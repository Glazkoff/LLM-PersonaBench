#!/bin/bash
#SBATCH --job-name=hbs-icml-serve
#SBATCH --partition=rocky
#SBATCH --account=proj_1759
#SBATCH --constraint=type_e|type_h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/lsavchenko/personality-arr/logs/icml_serve_%j.out
#SBATCH --error=/home/lsavchenko/personality-arr/logs/icml_serve_%j.err
# Serve $MODEL (dtype $DTYPE, TP $TP) and run $CMD with LOCAL_LLM_BASE_URL set. Used for transfer runs.
set -uo pipefail
export PATH=/home/lsavchenko/personality-arr/icmlenv/bin:$PATH
export HF_HOME=/home/lsavchenko/personality-arr/hf VLLM_LOGGING_LEVEL=WARNING TOKENIZERS_PARALLELISM=false
R=/home/lsavchenko/personality-arr/LLM-PersonaBench; cd $R
PY=/home/lsavchenko/personality-arr/icmlenv/bin/python; LOGDIR=/home/lsavchenko/personality-arr/logs
: "${MODEL:?}" "${CMD:?}"; TP=${TP:-1}; PORT=$((8800 + RANDOM % 150))
EXTRA=(); [ "${DTYPE:-bf16}" = "fp8" ] && EXTRA+=(--quantization fp8)
$PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT --host 127.0.0.1 --trust-remote-code \
  --tensor-parallel-size $TP --max-model-len 4096 --gpu-memory-utilization 0.90 --enable-prefix-caching \
  --max-logprobs 20 --max-num-seqs 512 "${EXTRA[@]}" > "$LOGDIR/vllm_serve_${SLURM_JOB_ID}.log" 2>&1 &
VPID=$!; trap 'kill $VPID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 $VPID 2>/dev/null || { echo "FATAL vLLM exited"; tail -30 "$LOGDIR/vllm_serve_${SLURM_JOB_ID}.log"; exit 3; }; sleep 10; done
export LOCAL_LLM_BASE_URL="http://127.0.0.1:$PORT/v1"
echo "running: $CMD"
eval "$PY $CMD"
rc=$?; echo "SERVE DONE rc=$rc"; exit $rc
