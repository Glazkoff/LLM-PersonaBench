#!/bin/bash
#SBATCH --job-name=hbs-h17pre
#SBATCH --partition=infer
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/home/glazkov/personality-twins-arr/logs/h17pre_%j.out
#SBATCH --error=/home/glazkov/personality-twins-arr/logs/h17pre_%j.err
# CPU-only, but still a Slurm job: nothing runs bare on this cluster.
# Checks (a) the corpora load and orientation is detected, (b) the scoring
# taxonomy holds, (c) which current-gen architectures this env can build.
set -uo pipefail
export HF_HOME=/home/glazkov/hf_cache
cd /home/glazkov/personality-twins-arr/LLM-PersonaBench
PY=/home/glazkov/personality-twins-arr/vllmenv/bin/python
"$PY" -c "import xlrd" 2>/dev/null || /home/glazkov/.local/bin/uv pip install --python "$PY" --quiet xlrd openpyxl

echo "########## scoring taxonomy ##########"
"$PY" arr2026/scripts/hyp/test_scoring.py 2>&1 | tail -14
echo "########## selection harness ##########"
"$PY" arr2026/scripts/hyp/test_h17_selection.py 2>&1 | tail -4
echo "########## orientation detector ##########"
"$PY" arr2026/scripts/hyp/test_orientation_detect.py 2>&1 | tail -4
echo "########## corpora ##########"
"$PY" - <<'PYEOF'
import sys; sys.path.insert(0,"arr2026/scripts/hyp")
from _corpora import load, check_orientation
for n in ("ipip","sd3","big5","hexaco"):
    d=check_orientation(load(n))
    print(f"{d['corpus']:7s} K={d['K']} n={d['n']:>6d} items={d['items']:>3d} "
          f"scales={d['scales']:>2d} recoded={d['detected_recoded']} rev={d['n_reverse']}")
PYEOF
echo "########## architectures ##########"
"$PY" - <<'PYEOF'
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
print("transformers", transformers.__version__)
for m in ["Qwen/Qwen3.8-27B","Qwen/Qwen3.6-35B-A3B","ibm-granite/granite-4.2-8b",
          "ibm-granite/granite-4.2-30b","google/gemma-4-12B-it","openai/gpt-oss-20b",
          "zai-org/GLM-4.7-Flash","swiss-ai/Apertus-8B-Instruct-2509"]:
    try:
        c=AutoConfig.from_pretrained(m, trust_remote_code=True)
        AutoModelForCausalLM._model_mapping[type(c)]
        print(f"  OK        {m:42s} {getattr(c,'model_type','?')}")
    except KeyError:
        print(f"  NO-ARCH   {m:42s} {getattr(c,'model_type','?')}")
    except Exception as e:
        print(f"  {type(e).__name__:9s} {m:42s} {str(e)[:40]}")
PYEOF
echo "PREFLIGHT DONE"
