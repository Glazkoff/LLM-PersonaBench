#!/bin/bash
#SBATCH --job-name=hbs-aamas-check
#SBATCH --partition=infer
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
set -euo pipefail
CAMPAIGN="$1"
CODE="$2"
PHASE="$3"
cd "$CODE"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export HBS_GIGAEVO_SOURCE="$CAMPAIGN/gigaevo-f72ab08"
export HBS_GIGAEVO_PYTHON="$CAMPAIGN/native-env/bin/python"
PY="$CAMPAIGN/env/bin/python"
sha256sum --check SOURCE_SHA256.txt > "$CAMPAIGN/logs/source-check-${SLURM_JOB_ID}.txt"
case "$PHASE" in
  test)
    test -f "$CAMPAIGN/ENV_READY"
    test -f "$CAMPAIGN/results-v3/READY.json"
    "$PY" -m pytest tests/optim -q
    "$PY" -m aamas2027.native_fixture --output "$CAMPAIGN/native-fixture-${SLURM_JOB_ID}"
    ;;
  external-download)
    "$PY" -m aamas2027.external download --revision f883165a3026fde855dfd448e0cd16443ab257b6 --out "$CAMPAIGN/twin-data"
    "$PY" -m aamas2027.external readiness --data "$CAMPAIGN/twin-data" > "$CAMPAIGN/TWIN_READINESS.json"
    ;;
  decoder-select)
    "$PY" -m aamas2027.decoders --stage select --old-root /home/glazkov/personality-twins-arr/LLM-PersonaBench --results "$CAMPAIGN/results-v3" --manifest aamas2027/manifest.json
    ;;
  freeze)
    test -f "$CAMPAIGN/results-v3/decoders/FROZEN.json"
    "$PY" -m aamas2027.analysis freeze --results "$CAMPAIGN/results-v3" --manifest aamas2027/manifest.json
    ;;
  decoder-eval)
    test -f "$CAMPAIGN/results-v3/FROZEN.json"
    "$PY" -m aamas2027.decoders --stage eval --old-root /home/glazkov/personality-twins-arr/LLM-PersonaBench --results "$CAMPAIGN/results-v3" --manifest aamas2027/manifest.json
    ;;
  analyze)
    "$PY" -m aamas2027.analysis analyze --results "$CAMPAIGN/results-v3" --manifest aamas2027/manifest.json
    ;;
  *) exit 2;;
esac
