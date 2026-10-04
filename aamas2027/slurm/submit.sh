#!/bin/bash
# Scheduler orchestration only; all experiment computation lives in finite jobs.
set -euo pipefail
CAMPAIGN="$1"
CODE="$2"
DATA_JOB="$3"
ENV_JOB="$4"
LEDGER="$CAMPAIGN/queue.tsv"
if test -s "$LEDGER"; then
  echo 'Existing queue ledger found; inspect it instead of duplicating the campaign.' >&2
  exit 2
fi
submit() {
  local key="$1"
  shift
  local id
  id=$(sbatch --parsable "$@")
  id=${id%%;*}
  printf '%s\t%s\n' "$key" "$id" >> "$LEDGER"
  printf '%s' "$id"
}
# Completed prerequisites may leave Slurm's controller cache before submission.
# 'receipt' is allowed only after their completed receipts have been verified;
# the CPU gate independently rechecks files/imports/tests/native execution.
DEP=()
PREREQ=""
for id in "$DATA_JOB" "$ENV_JOB"; do
  if test "$id" != receipt; then PREREQ="${PREREQ}:$id"; fi
done
if test -n "$PREREQ"; then DEP=(--dependency="afterok$PREREQ"); fi
CHECK=$(submit cpu-check "${DEP[@]}" --output="$CAMPAIGN/logs/check-%j.log" "$CODE/aamas2027/slurm/cpu.sbatch" "$CAMPAIGN" "$CODE" test)
EXTERNAL=$(submit external-download --dependency="afterok:$CHECK" --job-name=hbs-aamas-twin-data --time=00:10:00 --output="$CAMPAIGN/logs/twin-data-%j.log" "$CODE/aamas2027/slurm/cpu.sbatch" "$CAMPAIGN" "$CODE" external-download)
DSELECT=$(submit decoder-select --dependency="afterok:$CHECK" --job-name=hbs-aamas-dec-select --time=01:00:00 --output="$CAMPAIGN/logs/decoder-select-%j.log" "$CODE/aamas2027/slurm/cpu.sbatch" "$CAMPAIGN" "$CODE" decoder-select)
SMOKE=$(submit gpu-smoke --dependency="afterok:$CHECK" --array=0,42%1 --job-name=hbs-aamas-smoke --time=02:00:00 --output="$CAMPAIGN/logs/smoke-%A_%a.log" "$CODE/aamas2027/slurm/gpu.sbatch" "$CAMPAIGN" "$CODE" smoke)
SEARCH=$(submit search --dependency="afterok:$SMOKE" --array=0-83%2 --output="$CAMPAIGN/logs/search-%A_%a.log" "$CODE/aamas2027/slurm/gpu.sbatch" "$CAMPAIGN" "$CODE" search)
FREEZE=$(submit freeze --dependency="afterok:$SEARCH:$DSELECT" --job-name=hbs-aamas-freeze --time=00:10:00 --output="$CAMPAIGN/logs/freeze-%j.log" "$CODE/aamas2027/slurm/cpu.sbatch" "$CAMPAIGN" "$CODE" freeze)
FINAL=$(submit final --dependency="afterok:$FREEZE" --array=0-83%4 --gres=gpu:h200:1 --job-name=hbs-aamas-final --time=04:00:00 --output="$CAMPAIGN/logs/final-%A_%a.log" "$CODE/aamas2027/slurm/gpu.sbatch" "$CAMPAIGN" "$CODE" final)
DEVAL=$(submit decoder-eval --dependency="afterok:$FREEZE" --job-name=hbs-aamas-dec-eval --time=01:00:00 --output="$CAMPAIGN/logs/decoder-eval-%j.log" "$CODE/aamas2027/slurm/cpu.sbatch" "$CAMPAIGN" "$CODE" decoder-eval)
ANALYZE=$(submit analysis --dependency="afterok:$FINAL:$DEVAL" --job-name=hbs-aamas-analysis --time=01:00:00 --output="$CAMPAIGN/logs/analysis-%j.log" "$CODE/aamas2027/slurm/cpu.sbatch" "$CAMPAIGN" "$CODE" analyze)
cat "$LEDGER"
