#!/bin/bash
# Chain every remaining stage of the campaign on Euler behind the running Tier-1 arrays. Run ON EULER from the repo
# root: bash icml2027/scripts/pipeline.sh <tier1_job_id>[,<tier1_job_id>...]
#   Tier-1 -> E5 (NSGA-II) -> E8 (mutator sensitivity) -> stage: headline seeds -> transfers -> analysis
# Every array keeps %3, and each waits for the previous one, so the campaign never holds more than 3 cell GPUs
# plus the shared mutator.
set -euo pipefail
export PATH=/home/glazkov/personality-twins-arr/vllmenv/bin:$PATH
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
AFTER=$(echo "${1:?tier1 job ids}" | sed 's/,/:/g')
sub() { ICML_SITE=euler $PY icml2027/scripts/queue.py submit --grid "$1" --max-concurrent 3 --after "afterany:$2" \
        | grep -o "submitted [0-9]*" | awk '{print $2}' | tail -1; }
E5=$(sub icml2027/configs/grid_e5.yaml "$AFTER"); echo "E5=$E5"
E8=$(sub icml2027/configs/grid_e8.yaml "$E5"); echo "E8=$E8"
ST=$(sbatch --parsable --dependency=afterany:$E8 icml2027/slurm/stage.sbatch); echo "STAGE=$ST"
echo "tier1=$1 e5=$E5 e8=$E8 stage=$ST" > icml2027/queue/pipeline_jobs.txt
