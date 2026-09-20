#!/bin/bash
#SBATCH --job-name=hbs-h12
#SBATCH --partition=infer
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/h12_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/h12_%j.err
set -euo pipefail
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
"$PY" arr2026/scripts/hyp/h12_facet_locality.py \
  --results arr2026/results_euler \
  --runs h4_gptoss_20b h4_granite_8b h4_apertus_8b h4_qwen3_235b
echo "DONE rc=$?"
