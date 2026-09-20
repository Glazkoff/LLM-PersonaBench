#!/bin/bash
#SBATCH --job-name=hbs-e2b
#SBATCH --partition=infer
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/e2b_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/e2b_%j.err
set -euo pipefail
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
echo "=============== e2b head-to-head (per-item global baseline) ==============="
"$PY" arr2026/scripts/e2b_headtohead.py 2>&1 | tail -25 || echo "  (e2b FAILED)"
echo "=============== e3 (all 20 cells) ==============="
"$PY" arr2026/scripts/e3_metric_suite.py 2>&1 | tail -14 || echo "  (e3 FAILED)"
echo "DONE rc=$?"
