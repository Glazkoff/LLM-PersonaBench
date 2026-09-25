#!/bin/bash
#SBATCH --job-name=icml-analysis
#SBATCH --partition=infer
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/icml_analysis_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/icml_analysis_%j.err
# CPU only: baselines and split-noise floor, aggregation, bootstrap + BH, m* pricing, figures.
# Reads icml2027/results/, writes icml2027/results/{baselines,aggregates}. Deterministic.
set -euo pipefail
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
"$PY" -m src.optim.baselines --panels icml2027/configs/panels.yaml --out icml2027/results/baselines
"$PY" icml2027/scripts/analysis/aggregate.py --results icml2027/results --out icml2027/results/aggregates
"$PY" icml2027/scripts/analysis/bootstrap.py --results icml2027/results --panels icml2027/configs/panels.yaml --out icml2027/results/aggregates
"$PY" icml2027/scripts/analysis/price_mstar.py --results icml2027/results --panels icml2027/configs/panels.yaml --out icml2027/results/aggregates
"$PY" icml2027/scripts/analysis/figures.py --aggregates icml2027/results/aggregates --out icml2027/results/aggregates/figures
"$PY" icml2027/scripts/analysis/manifest.py --results icml2027/results
echo "ANALYSIS DONE"
