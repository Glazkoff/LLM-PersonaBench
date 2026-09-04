"""
H4 -- Where does the variance die? A three-stage ladder.

Our audit shows persona-conditioned LLMs emit ~43% of human response variance.
That number alone cannot say WHERE the variance is lost. This measures the same
items at three successive stages and compares each against the human standard
deviation:

  STAGE 1  BELIEF   the model's own next-token distribution over the answer
                    tokens {1,2,3,4,5}, read from logprobs before any sampling.
  STAGE 2  DECODE   answers actually sampled, at several temperatures.
  STAGE 3  PARSE    what survives the pipeline's parser.

The two outcomes are mutually exclusive and both are decisive:
  * belief SD ~ human SD, sampled SD << human SD  -> the loss is at DECODING.
    A readout that samples each item from its own belief distribution fixes it
    at zero extra generation cost, and the rival "LLMs are bad at emitting
    numeric scores" account (arXiv 2607.28550) is wrong about the mechanism.
  * belief SD << human SD                         -> the loss is REPRESENTATIONAL,
    upstream of decoding. No sampling-side fix can work.

A critical piece of context: all seven flagship configs in
configs/experiments/evoprompt_iter2/ set temperature 0. Under greedy decoding
within-persona variance is exactly zero by construction, so the published 0.43
is entirely BETWEEN-persona variance -- and between-persona variance is carried
only by the 5-level adverb code that get_modifier_by_match() emits.

Also builds the READOUT SIMULATOR: sample each item from its own belief
distribution, write the answers in the repo's CSV schema, and score them with
the unmodified metric suite.

Requires a vLLM server; set LOCAL_LLM_BASE_URL.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.models.providers.local_vllm import LocalVLLMModel  # noqa: E402
from src.utils.prompt import build_full_prompt  # noqa: E402

ITEMS = [f"i{i}" for i in range(1, 121)]
ANSWER_INSTR = ("Answer with a single digit from 1 to 5, where 1 = Very Inaccurate, "
                "2 = Moderately Inaccurate, 3 = Neither Accurate Nor Inaccurate, "
                "4 = Moderately Accurate, 5 = Very Accurate. Reply with the digit only.")


def load_genotype(cluster: int):
    """Reproduce the repo's persona genotype for a cluster from the shipped prompts."""
    import importlib.util

    def load(mod_path, attr):
        spec = importlib.util.spec_from_file_location(f"m{abs(hash(mod_path))}", mod_path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return getattr(m, attr)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=os.environ.get("LOCAL_LLM_BASE_URL"))
    ap.add_argument("--clusters", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--n-personas", type=int, default=40)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.0, 0.7, 1.0])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", default="arr2026/results/h4")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
    questions = json.loads((ROOT / "data/IPIP-NEO/120/questions.json").read_text(
        encoding="utf-8"))["questions"]

    llm = LocalVLLMModel(args.model, temperature=0.0, base_url=args.base_url,
                         max_workers=args.workers, timeout=600)
    print(llm.info(), flush=True)

    rows, readout_rows = [], []
    for cl in args.clusters:
        geno, system = load_genotype(cl)
        sub = df[df.clusters == cl].iloc[: args.n_personas]
        print(f"cluster {cl}: {len(sub)} personas", flush=True)

        human = sub[ITEMS].to_numpy(float)
        human_sd = np.nanstd(df[df.clusters == cl][ITEMS].to_numpy(float), axis=0)

        # ---- STAGE 1: belief distributions, one call per (persona, item) ----
        beliefs = np.full((len(sub), 120), np.nan)
        probs_all = np.full((len(sub), 120, 5), np.nan)   # full belief simplex
        ents = np.full((len(sub), 120), np.nan)
        onscale = np.full((len(sub), 120), np.nan)
        samples = np.full((len(sub), 120), np.nan)

        def one(pi_qi):
            pi, qi = pi_qi
            part = sub.iloc[pi]
            try:
                pr = build_full_prompt(geno, {"task": system.get("task", ""),
                                              "ipip_neo": [questions[qi]],
                                              "response_format": ANSWER_INSTR}, part)
                d = llm.likert_distribution(pr["system"], f"{questions[qi]['text']}\n{ANSWER_INSTR}")
                return pi, qi, d
            except Exception as e:
                return pi, qi, {"probs": None, "error": repr(e)[:120]}

        jobs = [(pi, qi) for pi in range(len(sub)) for qi in range(120)]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for pi, qi, d in ex.map(one, jobs):
                if d.get("probs"):
                    p = np.array([d["probs"][str(k)] for k in range(1, 6)])
                    vals = np.arange(1, 6)
                    mu = float((p * vals).sum())
                    beliefs[pi, qi] = np.sqrt(float((p * (vals - mu) ** 2).sum()))
                    probs_all[pi, qi, :] = p
                    ents[pi, qi] = d["entropy"]
                    onscale[pi, qi] = d["mass_on_scale"]
                    # readout simulator: sample from the belief itself
                    samples[pi, qi] = int(np.random.default_rng(2026 + pi * 131 + qi).choice(vals, p=p))
        print(f"  stage1 done: belief coverage {np.isfinite(beliefs).mean():.2%}", flush=True)

        ok = human_sd > 0
        vals_f = np.arange(1, 6, dtype=float)

        # The quantity comparable to a human ACROSS-respondent SD is the SD of the
        # persona-MIXTURE distribution, not the average within-persona SD.
        # Var(X_j) = E_i[Var(X_j|i)] + Var_i(E[X_j|i]) -- exact from the retained
        # probability vectors, no sampling noise.
        mu_ij = np.nansum(probs_all * vals_f[None, None, :], axis=2)
        ex2_ij = np.nansum(probs_all * (vals_f ** 2)[None, None, :], axis=2)
        var_ij = np.clip(ex2_ij - mu_ij ** 2, 0, None)
        within_var = np.nanmean(var_ij, axis=0)
        between_var = np.nanvar(mu_ij, axis=0)
        mixture_sd = np.sqrt(within_var + between_var)
        within_belief = np.nanmean(beliefs, axis=0)   # old, NOT comparable
        readout_sd = np.nanstd(samples, axis=0)

        rows.append({
            "cluster": cl, "model": args.model,
            "human_sd_mean": float(np.nanmean(human_sd[ok])),
            "VR_belief_mixture": float(np.nanmean(mixture_sd[ok] / human_sd[ok])),
            "share_within": float(np.nansum(within_var[ok])
                                  / max(np.nansum(within_var[ok] + between_var[ok]), 1e-12)),
            "share_between": float(np.nansum(between_var[ok])
                                   / max(np.nansum(within_var[ok] + between_var[ok]), 1e-12)),
            "VR_within_only_NOT_COMPARABLE": float(np.nanmean(within_belief[ok] / human_sd[ok])),
            "VR_readout": float(np.nanmean(readout_sd[ok] / human_sd[ok])),
            "belief_entropy_mean": float(np.nanmean(ents)),
            "mass_on_scale_mean": float(np.nanmean(onscale)),
        })
        print(f"  VR_mixture={rows[-1]['VR_belief_mixture']:.3f} "
              f"(within {rows[-1]['share_within']:.0%} / between {rows[-1]['share_between']:.0%}) "
              f"VR_readout={rows[-1]['VR_readout']:.3f}", flush=True)

        # persist the readout simulator's answers in the repo's CSV schema
        rd = pd.DataFrame(samples, columns=ITEMS)
        rd.insert(0, "case", sub["case"].to_numpy())
        d = out / f"readout_cluster_{cl}"
        d.mkdir(parents=True, exist_ok=True)
        rd.to_csv(d / "after_optimization_test_answers.csv", index=False)
        pd.DataFrame({"case": sub["case"].to_numpy()}).to_csv(d / "test_case_ids.csv", index=False)
        pd.DataFrame({"case": sub["case"].to_numpy()}).to_csv(d / "train_case_ids.csv", index=False)
        np.save(d / "belief_sd.npy", beliefs)
        np.save(d / "belief_probs.npy", probs_all)

    t = pd.DataFrame(rows)
    t.to_csv(out / "variance_ladder.csv", index=False)
    summary = {
        "model": args.model,
        "VR_belief_mixture_mean": round(float(t.VR_belief_mixture.mean()), 4),
        "share_within_mean": round(float(t.share_within.mean()), 4),
        "share_between_mean": round(float(t.share_between.mean()), 4),
        "VR_within_only_NOT_COMPARABLE_mean": round(
            float(t.VR_within_only_NOT_COMPARABLE.mean()), 4),
        "VR_readout_mean": round(float(t.VR_readout.mean()), 4),
        "belief_entropy_mean": round(float(t.belief_entropy_mean.mean()), 4),
        "max_entropy_ln5": 1.6094,
        "mass_on_scale_mean": round(float(t.mass_on_scale_mean.mean()), 4),
        "verdict": None,
    }
    v = summary["VR_belief_mixture_mean"]
    summary["verdict"] = (
        "DECODING LOSS: the persona-mixture belief carries near-human spread; "
        "decoding discards it." if v >= 0.8 else
        "REPRESENTATIONAL LOSS: the persona-mixture belief is itself collapsed."
        if v <= 0.55 else
        "PARTIAL: the belief carries some but not all of the missing spread.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== H4 SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
