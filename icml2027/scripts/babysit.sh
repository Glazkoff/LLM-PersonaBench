#!/bin/bash
# Babysit the icml2027 campaign on Euler. Idempotent; safe to run every 20-30 minutes.
#   1. reach Euler (direct, then via the AIRI jump host); exit 2 if unreachable
#   2. keep the shared mutator service alive (resubmit if no icml-mutator job exists)
#   3. for every grid in $GRIDS: if no icml-cells job is queued for it, resubmit unfinished cells
#      (queue.py is idempotent: completed/failed_readout/stuck cells are skipped, dead 'running' cells reset)
#   4. pull results back (JSON only; belief tensors stay on Euler), regenerate aggregates locally
#   5. commit the results and push the branch
# Prints one STATUS line at the end, for the loop to read.
set -uo pipefail
cd "$(dirname "$0")/../.."
LOCAL=$(pwd)
R=/home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
GRIDS=${GRIDS:-"icml2027/configs/grid_tier1.yaml"}
MAXC=${MAXC:-3}

HOST=""
for h in airi-h200 airi-h200-jump; do
  if timeout 45 ssh -o ConnectTimeout=25 -o BatchMode=yes "$h" true 2>/dev/null; then HOST=$h; break; fi
done
[ -z "$HOST" ] && { echo "STATUS unreachable"; exit 2; }
run() { timeout 300 ssh -o ConnectTimeout=25 -o BatchMode=yes "$HOST" "cd $R && $*"; }

# 2. mutator
if ! run "squeue -h -u glazkov -n icml-mutator -o %i" | grep -q .; then
  run "rm -f icml2027/queue/mutator.json; sbatch --parsable icml2027/slurm/mutator.sbatch" && echo "resubmitted mutator"
fi

# 3. cells
ACTIVE=$(run "squeue -h -u glazkov -n icml-cells -o %i" | wc -l | tr -d ' ')
if [ "$ACTIVE" = "0" ]; then
  for g in $GRIDS; do
    run "$PY icml2027/scripts/queue.py submit --grid $g --max-concurrent $MAXC --reset-running" | tail -1
  done
fi

# 4. pull results (small files only) and regenerate aggregates
rsync -az -e "ssh -o ConnectTimeout=25" --include='*/' --include='*.json' --include='*.jsonl' --include='*.csv' \
  --exclude='*' "$HOST:$R/icml2027/results/" "$LOCAL/icml2027/results/" 2>/dev/null
rsync -az -e "ssh -o ConnectTimeout=25" "$HOST:$R/icml2027/queue/" "$LOCAL/icml2027/queue/" 2>/dev/null
if ls icml2027/results/cells/*/eval_frozen.json >/dev/null 2>&1; then
  SKIP_BASELINES=1 PY="$LOCAL/.venv/bin/python" bash icml2027/scripts/analysis/regenerate_all.sh >/tmp/icml_regen.log 2>&1 \
    || echo "regenerate failed: $(tail -3 /tmp/icml_regen.log)"
fi

# 5. commit and push
COUNTS=$("$LOCAL/.venv/bin/python" icml2027/scripts/queue.py status 2>/dev/null | tail -1)
git add icml2027/results icml2027/queue >/dev/null 2>&1
if ! git diff --cached --quiet; then
  git commit -q -m "icml2027: results sync ($COUNTS)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin HEAD 2>/dev/null
fi
QUEUE=$(run "squeue -h -u glazkov -o '%j:%T'" | sort | uniq -c | tr '\n' ' ')
echo "STATUS host=$HOST $COUNTS | queue: $QUEUE"
