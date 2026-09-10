#!/bin/bash
# HSE compute nodes have no internet, so every checkpoint must be resident
# before any job starts. Run on the login node.
set -uo pipefail
export HF_HOME=/home/lsavchenko/personality-arr/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
PY=/home/lsavchenko/personality-arr/arrenv/bin/python
MODELS="
Qwen/Qwen3.8-27B
Qwen/Qwen3.6-35B-A3B
ibm-granite/granite-4.2-8b
ibm-granite/granite-4.2-30b
google/gemma-4-12B-it
openai/gpt-oss-20b
zai-org/GLM-4.7-Flash
swiss-ai/Apertus-8B-Instruct-2509
"
for M in $MODELS; do
  echo "=== $M ==="
  for try in 1 2 3; do
    "$PY" - "$M" <<'PYEOF' && break || { echo "  attempt $try failed, retrying"; sleep 30; }
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], max_workers=8,
                      ignore_patterns=["*.pth","*.gguf","original/*","consolidated*"])
print("  at", p, flush=True)
PYEOF
  done
done
echo "FETCH DONE"
du -sh /home/lsavchenko/personality-arr/hf
