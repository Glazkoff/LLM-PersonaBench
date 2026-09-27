#!/bin/bash
# Babysit the icml2027 campaign on two sites (Euler and HSE cHARISMa). Idempotent; run every 20-30 min.
# Per site: keep the shared mutator alive, resubmit unfinished cells of that site's grid when no cell job is
# queued, pull JSON results. Then regenerate aggregates locally, commit and push. Prints one STATUS line.
set -uo pipefail
cd "$(dirname "$0")/../.."
LOCAL=$(pwd)
MAXC=${MAXC:-3}
OUT=""

site() {  # name hosts remote_root python slurm_dir grid extra_sbatch_env
  local NAME=$1 HOSTS=$2 R=$3 PY=$4 SD=$5 GRIDS=$6 SITEENV=$7 HOST=""
  for h in $HOSTS; do
    if timeout 45 ssh -o ConnectTimeout=25 -o BatchMode=yes "$h" true 2>/dev/null; then HOST=$h; break; fi
  done
  [ -z "$HOST" ] && { OUT="$OUT | $NAME unreachable"; return; }
  run() { timeout 300 ssh -o ConnectTimeout=25 -o BatchMode=yes "$HOST" "cd $R && $*"; }
  local U; U=$(run 'whoami')
  # the shared mutator is needed only while cell arrays exist (pending or running); transfer and analysis jobs never
  # call it. Start it when cells are queued, cancel it when none are left, so it does not hold a GPU idle.
  local NCELLS; NCELLS=$(run "squeue -h -u $U -n hbs-icml-cells -o %i" | wc -l | tr -d ' ')
  local MUT; MUT=$(run "squeue -h -u $U -n hbs-icml-mutator -o %i" | head -1)
  if [ "$NCELLS" != "0" ] && [ -z "$MUT" ]; then
    run "rm -f icml2027/queue/mutator.json; sbatch --parsable $SD/mutator.sbatch" >/dev/null && OUT="$OUT | $NAME: resubmitted mutator"
  elif [ "$NCELLS" = "0" ] && [ -n "$MUT" ]; then
    run "scancel $MUT; rm -f icml2027/queue/mutator.json" && OUT="$OUT | $NAME: cancelled idle mutator $MUT"
  fi
  # throttle hand-off: 7461 ran at %1 while the first Tier-1 array (7388) finished; restore %3 once it is gone
  if [ "$NAME" = euler ] && ! run "squeue -h -j 7388 -o %i" | grep -q . && run "squeue -h -j 7461 -o %i" | grep -q .; then
    run "scontrol update jobid=7461 ArrayTaskThrottle=3" && OUT="$OUT | 7461 throttle -> 3"
  fi
  # retries: only when no campaign job of any kind is queued; unfinished cells of every grid are resubmitted as one
  # chain (each array waits for the previous), so the campaign still holds at most 3 cell GPUs
  if [ "$(run "squeue -h -u $U -o %j" | grep -c '^hbs-icml-\(cells\|stage\|transfer\|analysis\)')" = "0" ]; then
    local PREV=""
    for G in $GRIDS; do
      run "test -f $G" || continue
      local J
      J=$(run "$SITEENV $PY icml2027/scripts/queue.py submit --grid $G --max-concurrent $MAXC --reset-running ${PREV:+--after afterany:$PREV}" \
          | grep -o "submitted [0-9]*" | awk '{print $2}' | tail -1)
      [ -n "$J" ] && { PREV=$J; OUT="$OUT | $NAME: resubmitted $(basename $G .yaml) as $J"; }
    done
  fi
  for sub in cells crossmodel crosscluster; do
    rsync -az -e "ssh -o ConnectTimeout=25" --include='*/' --include='*.json' --include='*.jsonl' --include='*.csv' \
      --exclude='*' "$HOST:$R/icml2027/results/$sub/" "$LOCAL/icml2027/results/$sub/" 2>/dev/null
  done
  rsync -az -e "ssh -o ConnectTimeout=25" "$HOST:$R/icml2027/results/aggregates/ipip300_decoder_curve.json" \
    "$LOCAL/icml2027/results/aggregates/" 2>/dev/null
  rsync -az -e "ssh -o ConnectTimeout=25" "$HOST:$R/icml2027/configs/grid_headline.yaml" "$LOCAL/icml2027/configs/" 2>/dev/null
  rsync -az -e "ssh -o ConnectTimeout=25" "$HOST:$R/icml2027/queue/" "$LOCAL/icml2027/queue/$NAME/" 2>/dev/null
  OUT="$OUT | $NAME: $(run "squeue -h -u $U -o '%j:%T'" | grep '^hbs-icml' | sort | uniq -c | tr '\n' ' ')"
}

site euler "airi-h200 airi-h200-jump" /home/glazkov/personality-twins-arr/LLM-PersonaBench \
  /home/glazkov/personality-twins-arr/vllmenv/bin/python icml2027/slurm \
  "icml2027/configs/grid_tier1.yaml icml2027/configs/grid_e5.yaml icml2027/configs/grid_e8.yaml icml2027/configs/grid_headline.yaml" \
  "ICML_SITE=euler"
if [ "${HSE:-0}" = 1 ]; then
  site hse "hse" /home/lsavchenko/personality-arr/LLM-PersonaBench \
    /home/lsavchenko/personality-arr/icmlenv/bin/python icml2027/slurm/hse icml2027/configs/grid_tier1_hse.yaml "ICML_SITE=hse"
fi

if ls icml2027/results/cells/*/eval_frozen.json >/dev/null 2>&1; then
  SKIP_BASELINES=1 PY="$LOCAL/.venv/bin/python" bash icml2027/scripts/analysis/regenerate_all.sh >/tmp/icml_regen.log 2>&1 \
    || OUT="$OUT | regenerate failed: $(tail -2 /tmp/icml_regen.log | tr '\n' ' ')"
fi
COUNTS=$("$LOCAL/.venv/bin/python" icml2027/scripts/queue.py status 2>/dev/null | tail -1)
git add icml2027/results icml2027/queue >/dev/null 2>&1
if ! git diff --cached --quiet; then
  git commit -q -m "icml2027: results sync ($COUNTS)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin HEAD 2>/dev/null
fi
echo "STATUS $COUNTS$OUT"
