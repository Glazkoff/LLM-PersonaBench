#!/bin/bash
#SBATCH --job-name=icml-analysis
#SBATCH --partition=infer
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/icml_analysis_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/icml_analysis_%j.err
# CPU only. Baselines + aggregation + bootstrap/BH + m* + figures. SKIP_BASELINES=1 to reuse baselines.
set -euo pipefail
export PATH=/home/glazkov/personality-twins-arr/vllmenv/bin:$PATH
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python bash icml2027/scripts/analysis/regenerate_all.sh
