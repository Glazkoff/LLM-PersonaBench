#!/bin/bash
#SBATCH --job-name=hbs-icml-stage
#SBATCH --partition=infer
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/icml_stage_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/icml_stage_%j.err
# Runs after Tier-1, E5 and E8 have finished. Applies the pre-registered headline rule, then submits the
# headline-seed array, the transfer array (after it) and the final CPU analysis (after the transfers).
set -euo pipefail
export PATH=/home/glazkov/personality-twins-arr/vllmenv/bin:$PATH
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
$PY icml2027/scripts/analysis/select_headline.py
HL=$(ICML_SITE=euler $PY icml2027/scripts/queue.py submit --grid icml2027/configs/grid_headline.yaml --max-concurrent 3 \
     | grep -o "submitted [0-9]*" | awk '{print $2}' | tail -1)
DEP=${HL:+--dependency=afterany:$HL}
NT=$($PY -c "import yaml;print(sum(1 for m in yaml.safe_load(open('icml2027/configs/models.yaml'))['models'] if m.get('tier')==1))")
TR=$(sbatch --parsable $DEP --array=0-$((NT-1))%3 icml2027/slurm/transfer_array.sbatch)
AN=$(sbatch --parsable --dependency=afterany:$TR icml2027/slurm/cpu_analysis.sbatch)
echo "headline=$HL transfer=$TR analysis=$AN" | tee icml2027/queue/stage_jobs.txt
