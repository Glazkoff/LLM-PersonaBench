#!/bin/bash
#SBATCH --job-name=hbs-h25
#SBATCH --partition=infer
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=1-4
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/h25_%A_%a.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/h25_%A_%a.err
# H25: the wide incremental-value screen. CPU-ONLY by construction -- it is
# recomputed from saved belief tensors, so there is deliberately no --gres and
# it can never hold a GPU away from the readout jobs.
#
# RE-RUN under the hardened scorer (2026-09-18): NaN human answers no longer
# score as a top-category answer, refused forecasts no longer score perfectly,
# every arm is reduced over one common-support mask, and directories whose
# reverse-key count disagrees with the published key are refused.
# 191 run directories on this box carry the full artifact set. Sharded by
# family so a failure in one family does not cost the others.
#
# Sized for BACKFILL, not for comfort. Euler is a single node whose GPU jobs
# hold a reservation; an 8h/32-core CPU job sits behind that reservation on
# (Priority) even with 600 cores idle, because starting it would delay the
# reserved job. 2h/16 cores fits the backfill window and starts immediately.
# The local timing is ~2 min for 4 runs with the GBM arm, so the 110-run shard
# has ample headroom.
set -uo pipefail
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
"$PY" -c "import xlrd, sklearn" 2>/dev/null || \
  /home/glazkov/.local/bin/uv pip install --python "$PY" --quiet xlrd scikit-learn

case $SLURM_ARRAY_TASK_ID in
  1) G="h15_*"     ; T=h15 ;;
  2) G="h16_*"     ; T=h16 ;;
  3) G="h17_*"     ; T=h17 ;;
  4) G="h17n512*"  ; T=h17n512 ;;
  *) echo "no such task"; exit 2 ;;
esac

echo "=== h25 screen: $G on $(hostname), $(nproc) cores ==="
"$PY" arr2026/scripts/hyp/h25_incremental_screen.py \
  --roots arr2026/results_euler --glob "$G" \
  --out "arr2026/results_euler/h25_screen/${T}.csv"
rc=$?
echo "DONE rc=$rc"
exit $rc
