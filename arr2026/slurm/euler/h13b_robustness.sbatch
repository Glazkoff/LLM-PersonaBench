#!/bin/bash
#SBATCH --job-name=hbs-h13b
#SBATCH --partition=infer
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/h13b_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/h13b_%j.err
set -euo pipefail
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
"$PY" arr2026/scripts/hyp/h13b_oracle_robustness.py --boost \
  --results arr2026/results_euler \
  --runs h4_gptoss_20b h4_granite_8b h4_apertus_8b h4_qwen3_235b \
        h4_granite_8b_raw h4_apertus_8b_raw h4_gptoss_20b_raw
echo "DONE rc=$?"
