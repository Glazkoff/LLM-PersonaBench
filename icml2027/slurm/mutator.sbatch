#!/bin/bash
#SBATCH --job-name=hbs-icml-mutator
#SBATCH --partition=infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/icml_mutator_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/icml_mutator_%j.err
# Long-lived shared mutator server. Every cell job reads icml2027/queue/mutator.json for its URL,
# so all arms use the identical mutator. Single-node cluster: 127.0.0.1 is reachable from other jobs.
set -uo pipefail
export PATH=/home/glazkov/personality-twins-arr/vllmenv/bin:$PATH
export HF_HOME=/home/glazkov/hf_cache
export VLLM_LOGGING_LEVEL=WARNING
R=/home/glazkov/personality-twins-arr/LLM-PersonaBench
cd $R
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
MUT_ID=$($PY -c "import yaml;print(yaml.safe_load(open('icml2027/configs/models.yaml'))['mutator']['hf_id'])")
PORT=${MUT_PORT:-8791}
echo "mutator $MUT_ID on port $PORT node $(hostname)"
$PY -m vllm.entrypoints.openai.api_server --model "$MUT_ID" --port "$PORT" --host 127.0.0.1 --trust-remote-code \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --enable-prefix-caching --max-num-seqs 64 &
PID=$!
trap 'kill $PID 2>/dev/null; rm -f $R/icml2027/queue/mutator.json' EXIT
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "FATAL mutator exited"; exit 3; }
  sleep 10
done
printf '{"url": "http://127.0.0.1:%s/v1", "model": "%s", "job": "%s"}\n' "$PORT" "$MUT_ID" "$SLURM_JOB_ID" > $R/icml2027/queue/mutator.json
echo "mutator up"
wait $PID
