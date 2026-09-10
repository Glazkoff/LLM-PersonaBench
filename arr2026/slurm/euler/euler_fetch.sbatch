#!/bin/bash
#SBATCH --job-name=arr-h17fetch
#SBATCH --partition=infer
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/h17fetch_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/h17fetch_%j.err
# CPU-only, but a Slurm job: nothing runs bare on this cluster.
# Stages weights and PROVES completeness. On HSE the xet writer left 1MB and
# 31MB stubs behind, and earlier in this project a partially-downloaded
# checkpoint loaded fine and emitted garbage logits, which cost several GPU
# jobs to diagnose. So existence is not the test: every file the repo declares
# must be present at exactly the declared size before any GPU job trusts it.
set -uo pipefail
export HF_HOME=/home/glazkov/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
"$PY" /home/glazkov/personality-twins-arr/fetch_verify.py
