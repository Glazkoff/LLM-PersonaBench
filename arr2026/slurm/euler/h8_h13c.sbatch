#!/bin/bash
#SBATCH --job-name=arr-ci
#SBATCH --partition=infer
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/ci_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/ci_%j.err
set -euo pipefail
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
echo "=============== H8 recalibrated (40 vs 40) ==============="
"$PY" arr2026/scripts/hyp/h8_respondent_equivalents.py 2>&1 | tail -30 || echo "  (H8 FAILED)"
echo "=============== H13c bootstrap CIs on A ==============="
"$PY" arr2026/scripts/hyp/h13c_bootstrap.py --draws 2000 \
  --results arr2026/results_euler \
  --runs h4_gptoss_20b h4_granite_8b h4_apertus_8b h4_qwen3_235b \
        h4_granite_8b_raw h4_apertus_8b_raw h4_gptoss_20b_raw 2>&1 | tail -20 || echo "  (H13c FAILED)"
echo "DONE rc=$?"
