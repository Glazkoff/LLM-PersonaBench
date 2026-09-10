"""Can this environment actually build each candidate, and does it fit?

Checks the config only -- no weights -- so a model that transformers cannot
construct is caught on the login node instead of after a queue wait and a
multi-hundred-GB download.
"""
import sys
from transformers import AutoConfig, AutoModelForCausalLM
import transformers, torch

CANDIDATES = [
    ("Qwen/Qwen3.8-27B",                    56),
    ("Qwen/Qwen3.6-35B-A3B",                72),
    ("ibm-granite/granite-4.2-8b",          18),
    ("ibm-granite/granite-4.2-30b",         59),
    ("google/gemma-4-12B-it",               24),
    ("openai/gpt-oss-20b",                  42),
    ("swiss-ai/Apertus-v1.5-8B",            18),
    ("swiss-ai/Apertus-v1.5-70B",          144),
    ("mistralai/Mistral-Small-4-119B-2603", 239),
    ("zai-org/GLM-4.7-Flash",               58),
]
H100, H200 = 80, 141
print(f"transformers {transformers.__version__}  torch {torch.__version__}\n")
print(f"{'model':40s} {'arch':18s} {'GB':>5s} {'gpus':>12s}  status")
ok = []
for mid, gb in CANDIDATES:
    try:
        cfg = AutoConfig.from_pretrained(mid, trust_remote_code=True)
        mt = getattr(cfg, "model_type", "?")
        try:
            AutoModelForCausalLM._model_mapping[type(cfg)]
            status = "OK"
        except KeyError:
            status = "NO ARCH MAPPING"
    except Exception as e:
        mt, status = "-", f"{type(e).__name__}: {str(e)[:44]}"
    need = f"{max(1,-(-int(gb*1.2)//H200))}xH200 / {max(1,-(-int(gb*1.2)//H100))}xH100"
    print(f"{mid:40s} {mt:18s} {gb:>5d} {need:>12s}  {status}")
    if status == "OK":
        ok.append(mid)
print(f"\n{len(ok)}/{len(CANDIDATES)} loadable in this environment")
