#!/bin/bash
# Every aggregate from icml2027/results, deterministically. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-python}
if [ "${SKIP_BASELINES:-0}" != 1 ]; then
  $PY -m src.optim.baselines --panels icml2027/configs/panels.yaml --out icml2027/results/baselines
fi
$PY icml2027/scripts/analysis/price_mstar.py --results icml2027/results --panels icml2027/configs/panels.yaml --out icml2027/results/aggregates
$PY icml2027/scripts/analysis/aggregate.py --results icml2027/results --out icml2027/results/aggregates
$PY icml2027/scripts/analysis/bootstrap.py --results icml2027/results --panels icml2027/configs/panels.yaml --out icml2027/results/aggregates
$PY icml2027/scripts/analysis/figures.py --aggregates icml2027/results/aggregates --out icml2027/results/aggregates/figures
$PY icml2027/scripts/analysis/manifest.py --results icml2027/results
echo REGENERATED
