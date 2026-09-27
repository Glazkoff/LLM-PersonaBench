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
  local NAME=$1 HOSTS=$2 R=$3 PY=$4 SD=$5 GRID=$6 SITEENV=$7 HOST=""
  for h in $HOSTS; do
    if timeout 45 ssh -o ConnectTimeout=25 -o BatchMode=yes "$h" true 2>/dev/null; then HOST=$h; break; fi
  done
  [ -z "$HOST" ] && { OUT="$OUT | $NAME unreachable"; return; }
  run() { timeout 300 ssh -o ConnectTimeout=25 -o BatchMode=yes "$HOST" "cd $R && $*"; }
  local U; U=$(run 'whoami')
  if ! run "squeue -h -u $U -n icml-mutator -o %i" | grep -q .; then
    run "rm -f icml2027/queue/mutator.json; sbatch --parsable $SD/mutator.sbatch" >/dev/null && OUT="$OUT | $NAME: resubmitted mutator"
  fi
  if [ "$(run "squeue -h -u $U -n icml-cells -o %i" | wc -l | tr -d ' ')" = "0" ]; then
    run "$SITEENV $PY icml2027/scripts/queue.py submit --grid $GRID --max-concurrent $MAXC --reset-running" | tail -1 >/dev/null
  fi
  rsync -az -e "ssh -o ConnectTimeout=25" --include='*/' --include='*.json' --include='*.jsonl' --include='*.csv' \
    --exclude='*' "$HOST:$R/icml2027/results/cells/" "$LOCAL/icml2027/results/cells/" 2>/dev/null
  rsync -az -e "ssh -o ConnectTimeout=25" "$HOST:$R/icml2027/queue/" "$LOCAL/icml2027/queue/$NAME/" 2>/dev/null
  OUT="$OUT | $NAME: $(run "squeue -h -u $U -n icml-cells,icml-mutator -o '%j:%T'" | sort | uniq -c | tr '\n' ' ')"
}

site euler "airi-h200 airi-h200-jump" /home/glazkov/personality-twins-arr/LLM-PersonaBench \
  /home/glazkov/personality-twins-arr/vllmenv/bin/python icml2027/slurm icml2027/configs/grid_tier1_euler.yaml "ICML_SITE=euler"
if [ "${HSE:-1}" = 1 ]; then
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
