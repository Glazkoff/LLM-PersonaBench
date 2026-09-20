#!/bin/bash
#SBATCH --job-name=hbs-h4path
#SBATCH --partition=rocky
#SBATCH --qos=normal
#SBATCH -A proj_1759
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/home/lsavchenko/personality-twins-arr/logs/h4path_%j.out
#SBATCH --error=/home/lsavchenko/personality-twins-arr/logs/h4path_%j.err
# SAME-PATH replication: re-read on V100 (sm_70, fp16) -- the identical numeric
# path the original rows were measured on -- at the same n=40 panel. App. F now
# claims VR_between is robust ACROSS paths (fp16/sm_70 vs bf16/sm_90). This run
# is the control that claim needs: it separates a path effect from ordinary
# run-to-run drift within one path. If these reproduce the published values
# tightly, the cross-path deltas are attributable to the path.
#
# GPU SIZING. Request only what the model needs, and pass it on the command line:
#
#   Qwen/Qwen2.5-7B-Instruct                   15 GB  --gres=gpu:v100:1  BS=8
#   mistralai/Mistral-Small-24B-Instruct-2501  44 GB  --gres=gpu:v100:2  BS=4
#   Qwen/Qwen2.5-32B-Instruct                  62 GB  --gres=gpu:v100:3  BS=4
#   Qwen/QwQ-32B-Preview                       62 GB  --gres=gpu:v100:3  BS=4
#
# WALLTIME. The same readout script completes in 1:46-7:25 on Euler's H200s
# (sacct, h4-family jobs). V100 fp16 is several times slower and these models are
# larger, but 12h was still off by an order of magnitude -- and an over-long
# --time is not free: it disqualifies the job from backfilling into the gaps
# ahead of a higher-priority reservation, which is exactly what left a 1-GPU 7B
# job estimated 1.5 days out while a free device sat idle on cn-007. Request:
#   7B  --time=02:00:00     24B --time=04:00:00     32B --time=06:00:00
#
# The directive below is the safe default for the largest target; a smaller
# model MUST override it downward. This is not pedantry about tidiness -- on a
# 458-job queue no V100 node had four free devices, and a blanket gpu:v100:4
# request for the 7B model was scheduled six days out while a single free device
# sat on cn-007. Ask for four when you need four.
#
# GPU request (2026-09-21). This script previously carried NO --gres at all: it
# called nvidia-smi and loaded a multi-billion-parameter model on whatever GPUs
# happened to be visible on the node, without reserving any. That is a collision
# waiting to happen with whoever legitimately holds them. `gpu:v100:4` is the
# correct request and does three jobs at once -- it reserves the devices, it
# pins the run to the sm_70/fp16 path this experiment exists to measure, and it
# cannot land on a type_e a100 (which SIGILLs at python startup here), so no
# separate --constraint is needed. Four devices, not one: these are 32 GB cards
# and the targets are 24B-32B models (44-62 GB of weights), so a single V100
# cannot hold them. This matches hse_h4v100.sbatch, the variant that was sized
# correctly. --time drops 16h -> 12h to match it too.
set -uo pipefail
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export HF_HOME=/home/lsavchenko/imak-exp/.shared-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /home/lsavchenko/personality-twins-arr/LLM-PersonaBench

# `set -u` turns a forgotten --export into an unreadable failure, so say it plainly.
: "${MODEL:?set MODEL, e.g. sbatch --export=ALL,MODEL=Qwen/Qwen2.5-7B-Instruct,BS=8 hse_h4path.sbatch}"

# Preflight: fail loudly before doing any work. The first attempt at this queue
# allocated four V100 jobs and died on a missing 150MB corpus that is untracked
# in git and therefore absent from a fresh clone.
#
# The $f below was previously written as \$f -- the script had been created
# through an ssh heredoc, so the backslash reached disk and the test asked
# whether a file literally named '$f' existed. It never does, so the preflight
# failed unconditionally on four consecutive submissions while every file it
# checks was present, and the message could not name what it was missing.
for f in data/raw/df_ipipneo_120_clusters data/IPIP-NEO/120/questions.json \
         data/IPIP-NEO/120/item_key.csv src/prompt/system.json; do
  [ -e "$f" ] || { echo "PREFLIGHT FAIL: missing $f"; exit 2; }
done
echo "preflight: data ok"

PY=/home/lsavchenko/imak-exp/structured-extractor-train/.venv/bin/python

# GPU preflight. torch 2.6.0+cu124 should carry sm_70 kernels, but that has
# never actually been demonstrated here: every previous attempt died before
# touching the GPU, so "V100 works" is an assumption, not a measurement. Prove
# it with a real fp16 matmul in a few seconds rather than discovering it after
# a 12-hour allocation and a 62 GB model load.
"$PY" - <<'PYEOF' || { echo "PREFLIGHT FAIL: torch cannot use this GPU"; exit 3; }
import sys, torch
if not torch.cuda.is_available():
    print("  torch.cuda.is_available() is False", flush=True); sys.exit(1)
n = torch.cuda.device_count()
caps = [torch.cuda.get_device_capability(i) for i in range(n)]
print(f"  torch {torch.__version__} cuda {torch.version.cuda} | {n} device(s)", flush=True)
for i, c in enumerate(caps):
    print(f"    [{i}] {torch.cuda.get_device_name(i)} sm_{c[0]}{c[1]}", flush=True)
try:
    a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    print(f"  fp16 matmul ok, checksum finite: {bool(torch.isfinite(a @ a).all())}", flush=True)
except Exception as e:
    print(f"  fp16 matmul FAILED: {type(e).__name__}: {e}", flush=True); sys.exit(1)
PYEOF
echo "preflight: gpu ok"

M="$MODEL"; BS="${BS:-4}"
TAG=$(echo "$M" | tr '/.' '__' | tr -d ' ')
echo "=== H4-samepath $M (bs=$BS) on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
"$PY" arr2026/scripts/hyp/h4_variance_ladder_hf.py --model "$M" \
  --batch-size "$BS" --n-personas 40 --out "arr2026/results_v100/h4v100_${TAG}"
rc=$?; echo "DONE rc=$rc"; exit $rc
