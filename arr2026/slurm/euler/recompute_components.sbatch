#!/bin/bash
#SBATCH --job-name=arr-recomp
#SBATCH --partition=infer
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/recomp_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/recomp_%j.err
# CPU-only, but still a Slurm job: nothing runs bare on this cluster.
set -euo pipefail
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
# bare python3 on Euler has no numpy; the ARR work here uses vllmenv.
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
[ -x "$PY" ] || { echo "FATAL: $PY missing"; exit 1; }
"$PY" arr2026/scripts/hyp/h4_recompute_components.py \
  --results arr2026/results_euler --runs ${RUNS:-h4_qwen3_235b}
echo "DONE rc=$?"
