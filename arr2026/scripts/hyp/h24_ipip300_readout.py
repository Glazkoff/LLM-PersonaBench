r"""H24: same-person transfer to an unseen item family (IPIP-NEO-300).

Every disjoint result so far (h15/h16/h17) splits items WITHIN each scale: two
items of a facet condition the persona and the other two are targets. So every
target facet is still represented in the conditioning half, and a statistical
decoder always has a labelled sibling to borrow from. That is why the decoder
wins everywhere and why the LLM's incremental value measures zero (see
arr2026/strengthen/BRIEF.md sections 11 and 14).

`data/PAlign/Test-set.json` removes that crutch. 300 respondents answered the
full IPIP-NEO-300. 120 of those items are exactly the IPIP-NEO-120 short form
this paper conditions on everywhere else; the other 180 are items the same
person answered and no arm has ever been fitted on. Conditioning on the short
form and predicting the remaining 180 is genuine same-person, unseen-item
transfer, at ~300 respondents -- the lowest human-label regime available, which
is where a text-conditioned or LLM-based predictor has a reason to exist.

Two persona modes, because Track B needs both axes from one script:

  --persona scores   the profile lines used by h15/h17, so the readout is
                     comparable to every earlier result
  --persona history  M actual (item, answer) pairs drawn from the short form,
                     which is what an observed-history budget curve requires

Orientation: Test-set.json is stored RECODED -- verified against the published
`Sign` column, all 148 reverse-keyed items correlate positively with their
facet's forward-item mean, and detect_recoded returns True with mean_r 0.626.
The model answers the literal item, so 93 of the 180 targets must have their
simplex reversed before scoring. This script records which, and never applies
the flip itself: scoring code owns that, exactly as in h15_analyse.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
WORDS = ["very little", "slightly", "moderately", "quite strongly", "very strongly"]
LEVELS = ["very inaccurate", "moderately inaccurate", "neither accurate nor inaccurate",
          "moderately accurate", "very accurate"]
K = 5


def load_corpus():
    """Y (300,300) in Full# order, plus the published key and the 120/180 split."""
    kx = pd.read_excel(ROOT / "data/PAlign/IPIP-NEO-ItemKey.xls")
    full = kx["Full#"].astype(int).tolist()
    text = {int(r["Full#"]): str(r["Item"]).strip() for _, r in kx.iterrows()}
    facet = {int(r["Full#"]): str(r["Key"]).strip() for _, r in kx.iterrows()}
    rev = {int(r["Full#"]) for _, r in kx.iterrows()
           if str(r["Sign"]).strip().startswith("-")}
    short = {int(r["Full#"]) for _, r in kx.dropna(subset=["Short#"]).iterrows()}
    if len(full) != 300 or len(short) != 120:
        raise SystemExit(f"item key is malformed: {len(full)} items, {len(short)} short")

    rows = json.loads((ROOT / "data/PAlign/Test-set.json").read_text())
    Y = np.array([[float(r[f"i{j}"]) for j in range(1, 301)] for r in rows])
    Y[(Y < 1) | (Y > K)] = np.nan
    meta = pd.DataFrame([{k: r.get(k) for k in ("case", "sex", "age", "country")}
                         for r in rows])
    inp = sorted(short)
    tgt = sorted(set(range(1, 301)) - short)
    return Y, text, facet, rev, inp, tgt, meta


def facet_scores(Y, facet, inp):
    """Percentile-style score per facet from the conditioning items only.

    The corpus is stored recoded, so no flipping here -- flipping an already
    recoded corpus would un-recode it and describe the wrong person.
    """
    out, names = [], []
    for f in sorted({facet[i] for i in inp}):
        mem = [i - 1 for i in inp if facet[i] == f]
        blk = Y[:, mem]
        cnt = np.sum(~np.isnan(blk), axis=1)
        raw = np.where(cnt > 0, np.nansum(blk, axis=1) / np.maximum(cnt, 1), np.nan)
        out.append((raw - 1.0) / (K - 1.0) * 100.0)
        names.append(f)
    return np.column_stack(out), names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--persona", choices=["scores", "history"], default="scores")
    ap.add_argument("--n-observed", type=int, default=20,
                    help="history mode only: how many short-form answers the "
                         "prompt actually shows")
    ap.add_argument("--n-scored", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=260918)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug-dump", type=int, default=0)
    a = ap.parse_args()

    Y, text, facet, rev, inp, tgt, dmeta = load_corpus()
    S, names = facet_scores(Y, facet, inp)
    complete = np.where(~np.isnan(S).any(axis=1))[0]
    print(f"ipip300: {len(inp)} conditioning / {len(tgt)} target items, "
          f"{Y.shape[0]} respondents, {len(complete)} with a complete persona",
          flush=True)

    perm = complete[np.random.default_rng(a.seed).permutation(len(complete))]
    scored, train = perm[: a.n_scored], perm[a.n_scored:]
    if len(scored) < a.n_scored:
        raise SystemExit(f"only {len(scored)} complete respondents for a panel "
                         f"of {a.n_scored}")
    assert not (set(scored.tolist()) & set(train.tolist())), "panel/train overlap"
    print(f"  scored {len(scored)}, train {len(train)}", flush=True)

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

    def _load(cls):
        try:
            return cls.from_pretrained(a.model, dtype=dtype, **kw)
        except TypeError:
            return cls.from_pretrained(a.model, torch_dtype=dtype, **kw)

    try:
        model = _load(AutoModelForCausalLM)
    except ValueError:
        import transformers as _tf
        arch = (getattr(cfg, "architectures", None) or [None])[0]
        cls = getattr(_tf, arch, None) if arch else None
        if cls is None:
            raise
        print(f"  AutoModelForCausalLM rejected {type(cfg).__name__}; "
              f"loading {arch} directly", flush=True)
        model = _load(cls)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, K + 1)]
    if len(set(dids)) != K:
        raise SystemExit(f"answer tokens for 1..{K} are not distinct: {dids}")
    tmpl = getattr(tok, "chat_template", "") or ""
    prefill = ("<|channel|>final<|message|>My answer is "
               if "<|channel|>" in tmpl else "My answer is ")
    answer = (f"Answer with a single number from 1 (very inaccurate) to "
              f"{K} (very accurate). Answer:")

    # history mode draws each respondent's observed subset from a per-run
    # generator, so the same seed reproduces the same observations exactly.
    hrng = np.random.default_rng(a.seed + 1)
    observed = {}
    prompts = []
    for r in scored:
        if a.persona == "scores":
            lines = [f"- {nm} describes you {WORDS[min(int(S[r, k] / 20.0), 4)]}"
                     for k, nm in enumerate(names)]
            sysmsg = ("You are simulating a specific person. Their profile:\n"
                      + "\n".join(lines) + "\nAnswer as this person would.")
        else:
            avail = [i for i in inp if np.isfinite(Y[r, i - 1])]
            pick = sorted(hrng.choice(avail, min(a.n_observed, len(avail)),
                                      replace=False).tolist())
            observed[int(r)] = pick
            lines = [f'- "{text[i]}" -> {LEVELS[int(Y[r, i - 1]) - 1]}' for i in pick]
            sysmsg = ("You are simulating a specific person. Here is how they "
                      "answered some questions about themselves:\n"
                      + "\n".join(lines) + "\nAnswer the next question as this "
                      "person would.")
        for j in tgt:
            msgs = [{"role": "system", "content": sysmsg},
                    {"role": "user", "content": f"{text[j]}\n{answer}"}]
            try:
                t = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=False)
            except TypeError:
                t = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
            prompts.append(t + prefill)

    if a.debug_dump:
        print("=" * 72, flush=True)
        print(repr(prompts[0]), flush=True)
        with torch.no_grad():
            enc = tok(prompts[:1], return_tensors="pt", padding=True).to(model.device)
            lg = model(**enc).logits[:, -1, :].float()
        pr = torch.softmax(lg, dim=-1)[0]
        top = torch.topk(pr, 10)
        print("top10:", [(repr(tok.decode([i])), round(float(v), 4))
                         for v, i in zip(top.values, top.indices)], flush=True)
        print("digit mass", float(pr[dids].sum()), flush=True)
        if a.debug_dump == 1:
            raise SystemExit("debug dump only")

    P = np.full((len(scored), len(tgt), K), np.nan)
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
                if tot[k, 0] >= 0.1:
                    P[g // len(tgt), g % len(tgt)] = pr[k] / tot[k, 0]
            if b0 % (a.batch_size * 100) == 0:
                print(f"  {b0}/{len(prompts)}", flush=True)

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    ti = [j - 1 for j in tgt]
    ii = [i - 1 for i in inp]
    np.save(out / "belief_probs.npy", P)
    np.save(out / "target_answers.npy", Y[np.ix_(scored, ti)])
    np.save(out / "train_answers.npy", Y[np.ix_(train, ti)])
    # the raw conditioning answers, so every CPU baseline is reproducible with
    # no access to the corpus and no re-derivation of the split
    np.save(out / "observed_scored.npy", Y[np.ix_(scored, ii)])
    np.save(out / "observed_train.npy", Y[np.ix_(train, ii)])
    np.save(out / "code_test.npy", S[scored])
    np.save(out / "code_train.npy", S[train])
    mass = float(np.mean(np.concatenate(massv)))
    cov = float(np.isfinite(P).all(axis=2).mean())
    (out / "summary.json").write_text(json.dumps({
        "corpus": "ipip300", "model": a.model, "persona": a.persona,
        "n_observed": a.n_observed if a.persona == "history" else None,
        "K": K, "seed": a.seed,
        "input_items": len(inp), "target_items": len(tgt),
        "n_scored": len(scored), "n_train": len(train),
        "scored_idx": scored.tolist(), "train_idx": train.tolist(),
        "input_ids": inp, "target_ids": tgt,
        "facet_of_target": {str(j): facet[j] for j in tgt},
        "reverse_keyed_targets": sorted(j for j in tgt if j in rev),
        "corpus_recoded": True,
        "observed_items": {str(k): v for k, v in observed.items()} or None,
        "mass_on_scale_mean": round(mass, 4), "coverage": round(cov, 4),
        "valid": bool(mass >= 0.5 and cov >= 0.5),
    }, indent=2))
    print(json.dumps({"mass": mass, "coverage": cov,
                      "valid": bool(mass >= 0.5 and cov >= 0.5)}, indent=2))


if __name__ == "__main__":
    main()
