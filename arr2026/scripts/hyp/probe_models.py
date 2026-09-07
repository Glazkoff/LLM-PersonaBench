"""Find belief-measurable instruct models, by family, that this stack can load.

We do not guess model names. This queries the Hub for recent instruct/chat
models, keeps one or two per family we have not yet covered, and reports which
ones AutoConfig can actually build under the installed transformers. Only names
that survive here are worth a GPU slot.
"""
import argparse
import collections

from huggingface_hub import HfApi
from transformers import AutoConfig

COVERED = {"qwen2", "qwen3", "mistral", "glm4"}  # families already in the table


def family_of(model_id: str) -> str:
    org, _, name = model_id.partition("/")
    return name.lower().split("-")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--per-family", type=int, default=2)
    ap.add_argument("--max-b", type=float, default=35.0, help="max params (B) for one GPU")
    a = ap.parse_args()

    api = HfApi()
    # The Hub client keeps renaming/removing these kwargs; ask for the minimum
    # and sort locally.
    cands = list(api.list_models(pipeline_tag="text-generation",
                                 sort="downloads", limit=a.limit))
    cands.sort(key=lambda m: getattr(m, "downloads", 0) or 0, reverse=True)
    print(f"[hub returned {len(cands)} candidates]", flush=True)
    seen: dict[str, int] = collections.defaultdict(int)
    rows = []
    for m in cands:
        mid = m.id
        fam = family_of(mid)
        if any(c in mid.lower() for c in COVERED):
            continue
        if seen[fam] >= a.per_family:
            continue
        try:
            cfg = AutoConfig.from_pretrained(mid, trust_remote_code=False)
        except Exception as e:
            rows.append((mid, "CONFIG-FAIL", str(e)[:60]))
            continue
        seen[fam] += 1
        rows.append((mid, type(cfg).__name__, f"layers={getattr(cfg,'num_hidden_layers','?')} "
                                              f"hidden={getattr(cfg,'hidden_size','?')}"))
    print(f"[{len(rows)} rows after filtering; families kept: {dict(seen)}]", flush=True)
    for mid, status, detail in rows:
        print(f"{status:16s} {mid:52s} {detail}")


if __name__ == "__main__":
    main()
