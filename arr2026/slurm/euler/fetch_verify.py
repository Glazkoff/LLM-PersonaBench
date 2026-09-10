"""Stage checkpoints and PROVE they are complete before any job trusts them.

The xet/hf_transfer path failed here with "Internal Writer Error: Background
writer channel closed", leaving 31MB and 1MB stubs behind. Worse, this project
has already been bitten by the other outcome: a checkpoint that downloads far
enough to LOAD but produces garbage logits, which cost several GPU jobs to
diagnose. So existence is not the test -- every file the repo declares must be
present at exactly the declared size.
"""
import os, sys, time

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"   # the failing writer path
os.environ["HF_HUB_DISABLE_XET"] = "1"          # plain HTTPS: slower, reliable

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

SKIP = (".pth", ".gguf", ".msgpack", ".h5")
MODELS = [
    "Qwen/Qwen3.8-27B", "Qwen/Qwen3.6-35B-A3B",
    "ibm-granite/granite-4.2-8b", "ibm-granite/granite-4.2-30b",
    "google/gemma-4-12B-it", "openai/gpt-oss-20b",
    "zai-org/GLM-4.7-Flash", "swiss-ai/Apertus-8B-Instruct-2509",
]
api = HfApi()


def wanted(repo):
    """Files the repo declares, with their true sizes, minus formats we skip."""
    info = api.model_info(repo, files_metadata=True)
    return {s.rfilename: s.size for s in info.siblings
            if not s.rfilename.endswith(SKIP)
            and not s.rfilename.startswith(("original/", "consolidated"))
            and s.size is not None}


def missing(repo, local):
    out = []
    for name, size in wanted(repo).items():
        p = os.path.join(local, name)
        if not os.path.exists(p):
            out.append((name, size, "absent"))
        elif os.path.getsize(p) != size:
            out.append((name, size, f"truncated {os.path.getsize(p)}/{size}"))
    return out


for repo in MODELS:
    print(f"=== {repo} ===", flush=True)
    local = None
    for attempt in range(1, 6):
        try:
            local = snapshot_download(
                repo, max_workers=4,
                ignore_patterns=["*.pth", "*.gguf", "*.msgpack", "*.h5",
                                 "original/*", "consolidated*"])
        except Exception as e:
            print(f"  attempt {attempt}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            time.sleep(20); continue
        bad = missing(repo, local)
        if not bad:
            gb = sum(os.path.getsize(os.path.join(dp, f))
                     for dp, _, fs in os.walk(local) for f in fs) / 1e9
            print(f"  COMPLETE and verified, {gb:.1f}GB", flush=True)
            break
        print(f"  attempt {attempt}: {len(bad)} file(s) still incomplete, "
              f"e.g. {bad[0][0]} {bad[0][2]}", flush=True)
        time.sleep(20)
    else:
        print(f"  GIVING UP on {repo} -- do NOT run jobs against it", flush=True)
print("FETCH+VERIFY DONE", flush=True)
