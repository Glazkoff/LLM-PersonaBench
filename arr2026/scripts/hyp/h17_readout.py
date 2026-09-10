r"""Disjoint-item belief readout, for any instrument and any model.

Generalises h15 along three axes the extended study needs:

  * any corpus from _corpora (IPIP-120, SD3, IPIP-FFM-50, HEXACO)
  * any response-scale length K -- HEXACO is 7-point, so the answer tokens,
    the ANSWER instruction and every downstream score must follow the data
  * three disjoint respondent panels, so a model can be selected on one and
    evaluated on another it has never been scored against

The persona is built ONLY from each scale's input items and the readout covers
ONLY its target items, so the prompt cannot contain the answers being scored.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _corpora import load  # noqa: E402

WORDS = ["very little", "slightly", "moderately", "quite strongly", "very strongly"]


def split_items(ids, scale, n_input):
    inp, tgt = [], []
    for s in sorted({scale[i] for i in ids}):
        mem = [i for i in ids if scale[i] == s]
        inp += mem[:n_input]
        tgt += mem[n_input:]
    return sorted(inp), sorted(tgt)


def scores_from(c, inp):
    """Percentile-style score per scale, from the INPUT items only."""
    idx = {i: k for k, i in enumerate(c.ids)}
    out, names = [], []
    for s in sorted({c.scale[i] for i in inp}):
        mem = [i for i in inp if c.scale[i] == s]
        cols = []
        for i in mem:
            v = c.Y[:, idx[i]]
            # Flip only when the corpus is stored raw. Flipping an already
            # recoded corpus un-recodes it and builds the wrong persona.
            cols.append(v if c.recoded or i not in c.rev else (c.K + 1.0) - v)
        blk = np.stack(cols, axis=1)
        cnt = np.sum(~np.isnan(blk), axis=1)
        # A respondent who skipped every input item in this scale has no score
        # for it; NaN propagates into the persona line and is caught below,
        # rather than being averaged into a silent zero.
        raw = np.where(cnt > 0, np.nansum(blk, axis=1) / np.maximum(cnt, 1), np.nan)
        out.append((raw - 1.0) / (c.K - 1.0) * 100.0)
        names.append(s)
    return np.column_stack(out), names


def panels(eligible, a):
    """train / val / test2 -- disjoint by construction, verified before use.

    `eligible` is the subset of respondents with a complete persona; drawing
    from it keeps a respondent with a missing scale score from either killing
    the job mid-run or being handed a fabricated persona line.
    """
    eligible = np.asarray(eligible)
    perm = eligible[np.random.default_rng(a.seed).permutation(len(eligible))]
    train = perm[a.n_hold: a.n_hold + a.n_train]
    if a.panel == "train_only":
        raise SystemExit("train_only is not a scored panel")
    base = a.n_hold + len(train)
    off = {"val": 0, "test2": a.n_panel}[a.panel]
    scored = perm[base + off: base + off + a.n_panel]
    if len(scored) < a.n_panel:
        raise SystemExit(f"panel '{a.panel}' holds {len(scored)} of {a.n_panel}: "
                         f"{len(eligible)} eligible, {a.n_hold} reserved, "
                         f"{len(train)} training. Lower --n-train.")
    assert not (set(scored.tolist()) & set(train.tolist())), "panel/train overlap"
    return scored, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["ipip", "sd3", "big5", "hexaco"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--panel", default="val", choices=["val", "test2"])
    ap.add_argument("--n-panel", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-hold", type=int, default=256,
                    help="respondents reserved and never scored here")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=260910)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug-dump", type=int, default=0)
    a = ap.parse_args()

    c = load(a.corpus)
    inp, tgt = split_items(c.ids, c.scale, c.n_input)
    print(f"{c.name}: K={c.K} {len(inp)} input / {len(tgt)} target items, "
          f"{c.Y.shape[0]} respondents, recoded={c.recoded}", flush=True)
    S, names = scores_from(c, inp)
    complete = np.where(~np.isnan(S).any(axis=1))[0]
    dropped = c.Y.shape[0] - len(complete)
    if dropped:
        print(f"  {dropped} respondent(s) dropped: no score for at least one "
              f"scale from the input items", flush=True)
    scored, train = panels(complete, a)

    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    try:
        from transformers.integrations.finegrained_fp8 import FP8Experts
        FP8Experts._impl_tp_layer_overrides.setdefault(None, {})
    except Exception:
        pass
    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True)
    if getattr(cfg, "quantization_config", None):
        dtype = "auto"
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    kw = dict(device_map="auto", trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, **kw)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    digits = [str(d) for d in range(1, c.K + 1)]
    dids = [tok.encode(d, add_special_tokens=False)[0] for d in digits]
    if len(set(dids)) != c.K:
        raise SystemExit(f"answer tokens for 1..{c.K} are not distinct: {dids}")
    tmpl = getattr(tok, "chat_template", "") or ""
    prefill = ("<|channel|>final<|message|>My answer is "
               if "<|channel|>" in tmpl else "My answer is ")
    answer = (f"Answer with a single number from 1 (very inaccurate) to "
              f"{c.K} (very accurate). Answer:")

    prompts = []
    for r in scored:
        assert not np.isnan(S[r]).any(), "incomplete persona reached the panel"
        lines = [f"- {nm} describes you {WORDS[min(int(S[r, k] / 20.0), 4)]}"
                 for k, nm in enumerate(names)]
        sysmsg = ("You are simulating a specific person. Their profile:\n"
                  + "\n".join(lines) + "\nAnswer as this person would.")
        for j in tgt:
            msgs = [{"role": "system", "content": sysmsg},
                    {"role": "user", "content": f"{c.text[j]}\n{answer}"}]
            try:
                t = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=False)
            except TypeError:
                t = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
            prompts.append(t + prefill)

    if a.debug_dump:
        print("=" * 72, flush=True); print(repr(prompts[0]), flush=True)
        with torch.no_grad():
            enc = tok(prompts[:1], return_tensors="pt", padding=True).to(model.device)
            lg = model(**enc).logits[:, -1, :].float()
        pr = torch.softmax(lg, dim=-1)[0]
        top = torch.topk(pr, 10)
        print("top10:", [(repr(tok.decode([i])), round(float(v), 4))
                         for v, i in zip(top.values, top.indices)], flush=True)
        print("mass", float(pr[dids].sum()), flush=True)
        if a.debug_dump == 1:
            raise SystemExit("debug dump only")

    P = np.full((len(scored), len(tgt), c.K), np.nan)
    massv = []
    with torch.no_grad():
        for b0 in range(0, len(prompts), a.batch_size):
            enc = tok(prompts[b0: b0 + a.batch_size], return_tensors="pt",
                      padding=True).to(model.device)
            pr = torch.softmax(model(**enc).logits[:, -1, :].float(), dim=-1)
            pr = pr[:, dids].cpu().numpy()
            tot = pr.sum(axis=1, keepdims=True)
            massv.append(tot.ravel())
            for k in range(pr.shape[0]):
                g = b0 + k
                P[g // len(tgt), g % len(tgt)] = pr[k] / max(tot[k, 0], 1e-12)
            if b0 % (a.batch_size * 100) == 0:
                print(f"  {b0}/{len(prompts)}", flush=True)

    mass = float(np.mean(np.concatenate(massv)))
    cov = float(np.mean(np.concatenate(massv) >= 0.5))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "belief_probs.npy", P)
    np.save(out / "target_answers.npy", c.Y[np.ix_(scored, [c.ids.index(j) for j in tgt])])
    np.save(out / "train_answers.npy", c.Y[np.ix_(train, [c.ids.index(j) for j in tgt])])
    (out / "summary.json").write_text(json.dumps({
        "corpus": c.name, "model": a.model, "panel": a.panel, "K": c.K,
        "input_items": len(inp), "target_items": len(tgt),
        "n_scored": len(scored), "n_train": len(train),
        "n_dropped_incomplete": int(dropped),
        "mass_on_scale_mean": round(mass, 4), "coverage": round(cov, 4),
        "target_ids": tgt, "reverse_keyed_targets": sorted(j for j in tgt if j in c.rev),
        "corpus_recoded": c.recoded,
        "valid": bool(mass >= 0.5 and cov >= 0.5),
    }, indent=2))
    print(json.dumps({"mass": mass, "coverage": cov,
                      "valid": bool(mass >= 0.5 and cov >= 0.5)}, indent=2))


if __name__ == "__main__":
    main()
