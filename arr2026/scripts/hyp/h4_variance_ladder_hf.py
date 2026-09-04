"""
H4 (HF backend) -- belief-level variance measurement without a vLLM server.

Same measurement as h4_variance_ladder.py, but the model's distribution over the
answer tokens {1,2,3,4,5} is read straight from a forward pass with
transformers, rather than from a served OpenAI-compatible endpoint. On an older
cluster this is the more robust path: no server, no attention-backend
constraints, and the readout is exact rather than sampled.

For each (persona, item) we build the chat prompt, append an assistant prefix so
the very next token must be the answer digit, take the logits at the final
position, and renormalise over the five digit tokens. `mass_on_scale` records how
much probability landed on those five before renormalisation -- a low value means
the model wanted to say something else and the reading is unreliable.

Reports, per cluster:
  VR_belief_within  the model's own within-persona dispersion / human dispersion
  VR_readout        dispersion of a population sampled from those beliefs
  belief entropy    against ln(5) = 1.609, the maximum
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.utils.prompt import build_full_prompt  # noqa: E402

ITEMS = [f"i{i}" for i in range(1, 121)]
ANSWER_INSTR = ("Answer with a single digit from 1 to 5, where 1 = Very Inaccurate, "
                "2 = Moderately Inaccurate, 3 = Neither Accurate Nor Inaccurate, "
                "4 = Moderately Accurate, 5 = Very Accurate. Reply with the digit only.")
DIGITS = ["1", "2", "3", "4", "5"]


def load_genotype(cluster: int):
    base = ROOT / "src/prompt/mean_value_cluster"
    traits = json.loads((base / "traits.json").read_text(encoding="utf-8"))
    facets = json.loads((base / "facets.json").read_text(encoding="utf-8"))
    system = json.loads((ROOT / "src/prompt/system.json").read_text(encoding="utf-8"))
    ck = str(cluster)

    def pick(d):
        if ck in d:
            return d[ck]
        for k in (f"cluster_{cluster}", cluster):
            if k in d:
                return d[k]
        return d

    return {
        "role_definition": system.get("role_definition", "You are simulating a person."),
        "trait_formulations": pick(traits),
        "facet_formulations": pick(facets),
        "critic_formulations": system.get("critic_formulations", ""),
        "intensity_modifiers": {"boundaries": [0, 20, 40, 60, 80, 100],
                                "modifiers": ["very little", "slightly", "moderately",
                                              "quite strongly", "very strongly"]},
    }, system


def digit_token_ids(tok):
    """Token ids for the bare digits and for their space-prefixed variants."""
    ids = {}
    for d in DIGITS:
        cand = set()
        for form in (d, " " + d):
            e = tok.encode(form, add_special_tokens=False)
            if len(e) == 1:
                cand.add(e[0])
        if not cand:  # fall back to the first token of the digit
            cand = {tok.encode(d, add_special_tokens=False)[0]}
        ids[d] = sorted(cand)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--clusters", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--n-personas", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="arr2026/results/h4_hf")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
    questions = json.loads((ROOT / "data/IPIP-NEO/120/questions.json").read_text(
        encoding="utf-8"))["questions"]

    # V100 (Volta, capability 7.x) has no native bf16 -- asking for it there costs
    # roughly an order of magnitude in throughput. Pick fp16 on pre-Ampere cards.
    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    print(f"loading {args.model} ... (sm_{cap[0]}{cap[1]}, dtype={dtype})", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dids = digit_token_ids(tok)
    print("digit token ids:", dids, flush=True)

    rows = []
    for cl in args.clusters:
        geno, system = load_genotype(cl)
        sub = df[df.clusters == cl].iloc[: args.n_personas]
        human_all = df[df.clusters == cl][ITEMS].to_numpy(float)
        human_sd = np.nanstd(human_all, axis=0)
        print(f"cluster {cl}: {len(sub)} personas", flush=True)

        prompts = []
        for pi in range(len(sub)):
            part = sub.iloc[pi]
            pr = build_full_prompt(geno, {"task": system.get("task", ""),
                                          "ipip_neo": questions[:1],
                                          "response_format": ANSWER_INSTR}, part)
            for qi in range(120):
                msgs = [{"role": "system", "content": pr["system"]},
                        {"role": "user",
                         "content": f"{questions[qi]['text']}\n{ANSWER_INSTR}"}]
                try:
                    text = tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False)
                except TypeError:
                    text = tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                prompts.append(text + "My answer is ")

        beliefs = np.full((len(sub), 120), np.nan)
        probs_all = np.full((len(sub), 120, 5), np.nan)   # full belief simplex
        ents = np.full((len(sub), 120), np.nan)
        onscale = np.full((len(sub), 120), np.nan)
        samples = np.full((len(sub), 120), np.nan)
        rng = np.random.default_rng(args.seed + cl)
        vals = np.arange(1, 6)

        with torch.no_grad():
            for s in range(0, len(prompts), args.batch_size):
                chunk = prompts[s: s + args.batch_size]
                enc = tok(chunk, return_tensors="pt", padding=True,
                          truncation=True, max_length=4096).to(model.device)
                logits = model(**enc).logits[:, -1, :].float()
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                for b in range(len(chunk)):
                    idx = s + b
                    pi, qi = idx // 120, idx % 120
                    raw = np.array([probs[b, dids[d]].sum() for d in DIGITS])
                    tot = raw.sum()
                    if tot <= 0:
                        continue
                    p = raw / tot
                    mu = float((p * vals).sum())
                    beliefs[pi, qi] = float(np.sqrt((p * (vals - mu) ** 2).sum()))
                    probs_all[pi, qi, :] = p
                    ents[pi, qi] = float(-(p[p > 0] * np.log(p[p > 0])).sum())
                    onscale[pi, qi] = float(tot)
                    samples[pi, qi] = int(rng.choice(vals, p=p))
                if s % (args.batch_size * 50) == 0:
                    print(f"    {s}/{len(prompts)}", flush=True)

        ok = human_sd > 0
        vals_f = vals.astype(float)

        # The quantity that is comparable to a human ACROSS-respondent SD is the SD
        # of the persona-MIXTURE distribution, not the average within-persona SD.
        # Law of total variance:  Var(X_j) = E_i[Var(X_j|i)] + Var_i(E[X_j|i]).
        # Computed exactly from the retained probability vectors -- no sampling noise.
        mu_ij = np.nansum(probs_all * vals_f[None, None, :], axis=2)          # E[X|i,j]
        ex2_ij = np.nansum(probs_all * (vals_f ** 2)[None, None, :], axis=2)  # E[X^2|i,j]
        var_ij = np.clip(ex2_ij - mu_ij ** 2, 0, None)                        # Var(X|i,j)
        within_var = np.nanmean(var_ij, axis=0)          # E_i[Var(X_j|i)]
        between_var = np.nanvar(mu_ij, axis=0)           # Var_i(E[X_j|i])
        mixture_sd = np.sqrt(within_var + between_var)   # exact marginal SD
        within_sd_mean = np.nanmean(beliefs, axis=0)     # the old (incomparable) stat
        readout_sd = np.nanstd(samples, axis=0)          # 1-draw MC estimate of mixture

        rows.append({
            "model": args.model, "cluster": cl,
            "coverage": float(np.isfinite(beliefs).mean()),
            "human_sd_mean": float(np.nanmean(human_sd[ok])),
            # primary, comparable statistic
            "VR_belief_mixture": float(np.nanmean(mixture_sd[ok] / human_sd[ok])),
            "share_within": float(np.nansum(within_var[ok])
                                  / max(np.nansum(within_var[ok] + between_var[ok]), 1e-12)),
            "share_between": float(np.nansum(between_var[ok])
                                   / max(np.nansum(within_var[ok] + between_var[ok]), 1e-12)),
            # retained for transparency; NOT comparable to a human across-respondent SD
            "VR_within_only_NOT_COMPARABLE": float(np.nanmean(within_sd_mean[ok] / human_sd[ok])),
            "VR_readout": float(np.nanmean(readout_sd[ok] / human_sd[ok])),
            "belief_entropy_mean": float(np.nanmean(ents)),
            "mass_on_scale_mean": float(np.nanmean(onscale)),
        })
        print(f"  cluster {cl}: VR_mixture={rows[-1]['VR_belief_mixture']:.3f} "
              f"(within {rows[-1]['share_within']:.0%} / between {rows[-1]['share_between']:.0%}) "
              f"VR_readout={rows[-1]['VR_readout']:.3f} "
              f"entropy={rows[-1]['belief_entropy_mean']:.3f}", flush=True)

        d = out / f"readout_cluster_{cl}"
        d.mkdir(parents=True, exist_ok=True)
        rd = pd.DataFrame(samples, columns=ITEMS)
        rd.insert(0, "case", sub["case"].to_numpy())
        rd.to_csv(d / "after_optimization_test_answers.csv", index=False)
        for name in ("test_case_ids.csv", "train_case_ids.csv"):
            pd.DataFrame({"case": sub["case"].to_numpy()}).to_csv(d / name, index=False)
        np.save(d / "belief_sd.npy", beliefs)
        np.save(d / "belief_probs.npy", probs_all)

    t = pd.DataFrame(rows)
    t.to_csv(out / "variance_ladder.csv", index=False)
    v = float(t.VR_belief_mixture.mean())
    summary = {
        "model": args.model,
        "VR_belief_mixture_mean": round(v, 4),
        "share_within_mean": round(float(t.share_within.mean()), 4),
        "share_between_mean": round(float(t.share_between.mean()), 4),
        "VR_within_only_NOT_COMPARABLE_mean": round(
            float(t.VR_within_only_NOT_COMPARABLE.mean()), 4),
        "VR_readout_mean": round(float(t.VR_readout.mean()), 4),
        "belief_entropy_mean": round(float(t.belief_entropy_mean.mean()), 4),
        "max_entropy_ln5": 1.6094,
        "mass_on_scale_mean": round(float(t.mass_on_scale_mean.mean()), 4),
        "coverage": round(float(t.coverage.mean()), 4),
        "verdict": ("DECODING LOSS: the persona-mixture belief carries near-human spread; "
                    "sampling discards it." if v >= 0.8 else
                    "REPRESENTATIONAL LOSS: the persona-mixture belief is itself collapsed."
                    if v <= 0.55 else "PARTIAL"),
        "note": ("VR_belief_mixture is the SD of the persona-mixture distribution "
                 "(law of total variance), which is the quantity comparable to a human "
                 "across-respondent SD. VR_within_only is retained only to show the "
                 "earlier, non-comparable statistic."),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== H4-HF SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
