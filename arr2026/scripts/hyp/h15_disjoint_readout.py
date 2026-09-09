r"""H15: belief readout under a DISJOINT conditioning/target item split.

Every belief result so far conditions on scores computed from the same items it
is then scored on, so a sceptic can say the models are reconstructing arithmetic
rather than predicting a person. This removes that objection by construction.

Items are split within each scale. The persona prompt is built ONLY from the
input half; the readout covers ONLY the target half. No cluster label is used --
the released IPIP clustering is derived from all 120 items and would reimport
target information through the back door.

  IPIP-NEO-120: 2 input + 2 target items per facet -> 60 input, 60 target
  SD3:          4 input + 5 target items per trait -> 12 input, 15 target

Scores, bin thresholds and every baseline are computed on training respondents
only; the evaluated respondents' target answers are never seen before scoring.
The elicitation is byte-identical to the main experiment: thinking disabled, the
assistant turn prefilled, and the belief read from the digit logprobs.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
DIGITS = ["1", "2", "3", "4", "5"]
BOUNDS = [0, 20, 40, 60, 80, 100]
WORDS = ["very little", "slightly", "moderately", "quite strongly", "very strongly"]
ANSWER = ("Answer with a single number from 1 (very inaccurate) to "
          "5 (very accurate). Answer:")


def load_ipip():
    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
    key = pd.read_csv(ROOT / "data/IPIP-NEO/120/item_key.csv")
    text = {int(r.item): r.text for r in key.itertuples()}
    scale = {int(r.item): r.facet_key for r in key.itertuples()}
    rev = {int(r.item) for r in key.itertuples() if str(r.reverse).lower() == "true"}
    cols = [f"i{i}" for i in range(1, 121)]
    # corpus stores RECODED answers (trait-oriented already)
    return df[cols].to_numpy(float), list(range(1, 121)), text, scale, rev, 2, True


def load_sd3():
    rows = list(csv.DictReader(open(ROOT / "data/PAlign/Dark-Triad.csv"), delimiter="\t"))
    keys = [f"{t}{i}" for t in "MNP" for i in range(1, 10)]
    Y = np.array([[float(r[k]) for k in keys] for r in rows])
    Y = Y[(Y >= 1).all(axis=1)]
    kx = pd.read_excel(ROOT / "data/PAlign/dark_triad-ItemKey.xls")
    text = {i + 1: str(r.Item).strip() for i, r in enumerate(kx.itertuples())}
    scale = {i + 1: keys[i][0] for i in range(len(keys))}
    rev = {i + 1 for i, r in enumerate(kx.itertuples()) if str(r.Sign).strip() == "-"}
    # corpus stores RAW answers: reverse items correlate -0.19..-0.31 with
    # their trait's forward items, so they have not been recoded
    return Y, list(range(1, 28)), text, scale, rev, 4, False


def split_items(ids, scale, n_input):
    """Within each scale, the first n_input items condition; the rest are targets."""
    inp, tgt = [], []
    for s in sorted(set(scale.values())):
        mem = [i for i in ids if scale[i] == s]
        inp += mem[:n_input]
        tgt += mem[n_input:]
    return sorted(inp), sorted(tgt)


def scores_from(Y, ids, inp, scale, rev, recoded):
    """Percentile-style score per scale, from the INPUT items only."""
    idx = {i: k for k, i in enumerate(ids)}
    out, names = [], []
    for s in sorted(set(scale.values())):
        mem = [i for i in inp if scale[i] == s]
        if not mem:
            continue
        cols = []
        for i in mem:
            v = Y[:, idx[i]]
            # flip only when the corpus is raw; flipping an already-recoded
            # corpus would un-recode it and describe the wrong profile
            cols.append(v if recoded or i not in rev else 6.0 - v)
        raw = np.mean(cols, axis=0)
        out.append((raw - 1.0) / 4.0 * 100.0)           # 1..5 -> 0..100
        names.append(s)
    return np.column_stack(out), names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", choices=["ipip", "sd3"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-test", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=260909)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    Y, ids, text, scale, rev, n_in, recoded = (
        load_ipip() if a.instrument == "ipip" else load_sd3())
    inp, tgt = split_items(ids, scale, n_in)
    print(f"{a.instrument}: {len(inp)} input items, {len(tgt)} target items, "
          f"{len(Y)} respondents", flush=True)

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(Y))
    test, train = perm[: a.n_test], perm[a.n_test: a.n_test + a.n_train]
    S, names = scores_from(Y, ids, inp, scale, rev, recoded)

    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
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
    dids = [tok.encode(d, add_special_tokens=False)[0] for d in DIGITS]
    tmpl = getattr(tok, "chat_template", "") or ""
    prefill = ("<|channel|>final<|message|>My answer is "
               if "<|channel|>" in tmpl else "My answer is ")

    prompts = []
    for r in test:
        lines = []
        for k, nm in enumerate(names):
            b = min(int(S[r, k] / 20.0), 4)
            lines.append(f"- {nm} describes you {WORDS[b]}")
        sysmsg = ("You are simulating a specific person. Their profile:\n"
                  + "\n".join(lines)
                  + "\nAnswer as this person would.")
        for j in tgt:
            msgs = [{"role": "system", "content": sysmsg},
                    {"role": "user", "content": f"{text[j]}\n{ANSWER}"}]
            try:
                t = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(t + prefill)

    P = np.full((len(test), len(tgt), 5), np.nan)
    massv = []
    with torch.no_grad():
        for b0 in range(0, len(prompts), a.batch_size):
            chunk = prompts[b0: b0 + a.batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
            logits = model(**enc).logits[:, -1, :].float()
            pr = torch.softmax(logits, dim=-1)[:, dids].cpu().numpy()
            tot = pr.sum(axis=1, keepdims=True)
            massv.append(tot.ravel())
            for k in range(len(chunk)):
                g = b0 + k
                if tot[k, 0] >= 0.1:
                    P[g // len(tgt), g % len(tgt)] = pr[k] / tot[k, 0]
            if b0 % (a.batch_size * 100) == 0:
                print(f"  {b0}/{len(prompts)}", flush=True)

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "belief_probs.npy", P)
    np.save(out / "target_answers.npy", Y[np.ix_(test, [ids.index(j) for j in tgt])])
    np.save(out / "train_answers.npy", Y[np.ix_(train, [ids.index(j) for j in tgt])])
    np.save(out / "code_test.npy", S[test])
    np.save(out / "code_train.npy", S[train])
    mass = float(np.nanmean(np.concatenate(massv)))
    cov = float(np.isfinite(P).all(axis=2).mean())
    (out / "summary.json").write_text(json.dumps({
        "instrument": a.instrument, "model": a.model,
        "input_items": len(inp), "target_items": len(tgt),
        "n_test": len(test), "n_train": len(train),
        "mass_on_scale_mean": round(mass, 4), "coverage": round(cov, 4),
        "target_ids": tgt,
        "reverse_keyed_targets": sorted(j for j in tgt if j in rev),
        "corpus_recoded": recoded,
        "valid": bool(mass >= 0.5 and cov >= 0.5),
    }, indent=2))
    print(json.dumps({"mass": mass, "coverage": cov,
                      "valid": bool(mass >= 0.5 and cov >= 0.5)}, indent=2))


if __name__ == "__main__":
    main()
