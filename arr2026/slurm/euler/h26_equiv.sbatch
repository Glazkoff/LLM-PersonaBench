#!/bin/bash
#SBATCH --job-name=hbs-h26
#SBATCH --partition=infer
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=1-4
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/h26_%A_%a.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/h26_%A_%a.err
# H26: the answer-equivalent m* for every saved cell. CPU-ONLY -- recomputed
# from saved belief tensors, deliberately no --gres.
#
# Sized for backfill (2h / 16 cores): Euler is one node whose GPU jobs hold a
# reservation, and a longer CPU job sits on (Priority) behind it even with 600
# cores idle.
#
# Breadth is the point. Three cells is not enough to claim m*=1 is a property of
# persona prompting rather than of one weak prompt, so every model and every
# corpus in the artifact set is priced on the same scale.
set -uo pipefail
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
"$PY" -c "import xlrd, sklearn" 2>/dev/null || \
  /home/glazkov/.local/bin/uv pip install --python "$PY" --quiet xlrd scikit-learn

R=arr2026/results_euler
case $SLURM_ARRAY_TASK_ID in
  1) RUNS=$(ls -d $R/h15_* $R/h16_* 2>/dev/null); T=h15h16 ;;
  2) RUNS=$(ls -d $R/h17_[a-i]* 2>/dev/null);     T=h17a ;;
  3) RUNS=$(ls -d $R/h17_[j-z]* 2>/dev/null);     T=h17b ;;
  4) RUNS=$(ls -d $R/h17_[A-Z]* $R/h24_* 2>/dev/null); T=h17c_h24 ;;
  *) echo "no such task"; exit 2 ;;
esac
N=$(echo "$RUNS" | grep -c . || true)
echo "=== h26 shard $T: $N runs on $(hostname) ==="
[ "$N" -eq 0 ] && { echo "nothing to do"; exit 0; }

"$PY" arr2026/scripts/hyp/h26_answer_equivalent.py \
  --runs $RUNS --reps 5 \
  --out "$R/h26_equiv/${T}.json"
rc=$?
echo "DONE rc=$rc"
exit $rc
