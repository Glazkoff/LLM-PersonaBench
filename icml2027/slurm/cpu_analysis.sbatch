#!/bin/bash
#SBATCH --job-name=hbs-icml-analysis
#SBATCH --partition=infer
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/icml_analysis_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/icml_analysis_%j.err
# CPU only, on Euler where the belief tensors live: C2ST for every cell, then baselines, m*, aggregation,
# bootstrap/BH, hypotheses H1-H6, RESULTS_BRIEF.md, figures, manifest.
set -euo pipefail
export PATH=/home/glazkov/personality-twins-arr/vllmenv/bin:$PATH
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
$PY icml2027/scripts/analysis/c2st.py --results icml2027/results --panels icml2027/configs/panels.yaml
PY=$PY bash icml2027/scripts/analysis/regenerate_all.sh
echo "ANALYSIS DONE"
