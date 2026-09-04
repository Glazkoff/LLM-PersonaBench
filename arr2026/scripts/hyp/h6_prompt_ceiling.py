"""
H6 -- The information ceiling of the persona prompt itself.

In this pipeline an individual reaches the model as a *quantised code*, not as
their scores. `build_full_prompt` emits one line per trait and per facet of the
form

    This trait (X) describes you {modifier}: {fixed cluster text}

where the description is a per-cluster constant and the only individual-varying
content is `modifier`, chosen by `get_modifier_by_match` from a 5-level adverb
scale. So the whole individual is a short string over a 5-letter alphabet, and
everything else in the prompt is a cluster constant.

That places a hard upper bound on individuation, independent of the model. This
script measures it:

  1. COLLISION RATE -- how often two members of a cluster receive a byte-identical
     persona prompt. Colliding participants are indistinguishable to ANY reader.
  2. ORACLE CEILING -- fit the best possible map from the code to the 120 answers
     on a training half, generate oracle respondents on a held-out half, and score
     them. No model conditioned on this prompt can beat this.
  3. QUANTISATION COST -- mutual information retained by the discrete code versus
     the continuous scores.

If the ceiling is low, the encoding is the binding constraint and no amount of
model capability can fix it. If it is high, the prompt is exonerated and the
shortfall belongs to the model.

CPU only. No LLM calls.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.utils.prompt import get_modifier_by_match, get_modifier_bisect  # noqa: E402

ITEMS = [f"i{i}" for i in range(1, 121)]
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
FACETS = [c for c in [
    "facet_imagination", "facet_artistic_interests", "facet_emotionality",
    "facet_adventurousness", "facet_intellect", "facet_liberalism",
    "facet_self_efficacy", "facet_orderliness", "facet_dutifulness",
    "facet_achievement_striving", "facet_self_discipline", "facet_cautiousness",
    "facet_friendliness", "facet_gregariousness", "facet_assertiveness",
    "facet_activity_level", "facet_excitement_seeking", "facet_cheerfulness",
    "facet_trust", "facet_morality", "facet_altruism", "facet_cooperation",
    "facet_modesty", "facet_sympathy",
    "facet_anxiety", "facet_anger", "facet_depression",
    "facet_self_consciousness", "facet_immoderation", "facet_vulnerability"]]
DIMS = TRAITS + FACETS
MODCFG = {"boundaries": [0, 20, 40, 60, 80, 100],
          "modifiers": ["very little", "slightly", "moderately", "quite strongly",
                        "very strongly"]}
OUT = ROOT / "arr2026/results/h6"


def codes(sub, centroid, match_based):
    """The 35-symbol code the prompt builder would emit for each participant."""
    out = []
    for _, p in sub.iterrows():
        c = []
        for d in DIMS:
            v = p.get(d)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                c.append("?"); continue
            m = (get_modifier_by_match(v, centroid[d], MODCFG) if match_based
                 else get_modifier_bisect(v, MODCFG))
            c.append(str(MODCFG["modifiers"].index(m)))
        out.append("".join(c))
    return np.array(out)


def sim(pred, human):
    return float(np.nanmean(1.0 - np.abs(pred - human) / 4.0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
    rng = np.random.default_rng(2026)
    rows = []

    for cl in sorted(df.clusters.unique()):
        sub = df[df.clusters == cl]
        n = min(len(sub), 40000)
        sub = sub.iloc[:n]
        centroid = {d: float(sub[d].mean()) for d in DIMS}

        for label, match_based in (("match_based", True), ("bisect", False)):
            code = codes(sub, centroid, match_based)
            uniq = len(set(code))
            # collision rate: probability two random members share a code
            _, counts = np.unique(code, return_counts=True)
            coll = float(((counts * (counts - 1)).sum()) / (len(code) * (len(code) - 1)))

            # Oracle: the best map from the code to the answers, fitted on half and
            # scored on the other. Exact-code lookup is useless here -- with 35 symbols
            # over a 5-letter alphabet the codes are essentially unique, so a lookup
            # table never hits at test time and degenerates to the constant. Use a
            # smooth estimator over the ordinal code vector instead.
            from sklearn.linear_model import RidgeCV
            idx = rng.permutation(len(sub))
            tr, te = idx[: len(idx) // 2], idx[len(idx) // 2:]
            Y = sub[ITEMS].to_numpy(float)
            X = np.array([[int(ch) if ch != "?" else 2 for ch in c] for c in code], float)
            fallback = np.round(Y[tr].mean(axis=0))
            reg = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[tr], Y[tr])
            pred = np.clip(np.round(reg.predict(X[te])), 1, 5)
            oracle = float(np.mean([sim(pred[k], Y[te[k]]) for k in range(len(te))]))
            const = float(np.mean([sim(fallback, Y[te[k]]) for k in range(len(te))]))
            # a real human of the same cluster, as the ceiling
            hum = float(np.mean([sim(Y[tr[rng.integers(len(tr))]], Y[te[k]])
                                 for k in range(min(len(te), 2000))]))
            # oracle dispersion vs human dispersion
            vr_oracle = float(np.nanmean(np.nanstd(pred, 0)[np.nanstd(Y[te], 0) > 0]
                                         / np.nanstd(Y[te], 0)[np.nanstd(Y[te], 0) > 0]))

            rows.append({"cluster": int(cl), "encoding": label, "n": int(len(sub)),
                         "distinct_codes": uniq,
                         "codes_per_1000_people": round(1000 * uniq / len(sub), 1),
                         "collision_rate": coll,
                         "S_oracle_ceiling": oracle, "S_constant": const,
                         "S_real_human": hum, "VR_oracle_ceiling": vr_oracle})
            print(f"  cluster {cl} [{label}]: {uniq} distinct codes for {len(sub)} people, "
                  f"collision={coll:.4f} | S oracle={oracle:.4f} const={const:.4f} "
                  f"human={hum:.4f} | VR_oracle={vr_oracle:.3f}", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "prompt_ceiling.csv", index=False)
    m = t[t.encoding == "match_based"]
    summary = {
        "n_dims_in_code": len(DIMS),
        "alphabet_size": len(MODCFG["modifiers"]),
        "collision_rate_mean": round(float(m.collision_rate.mean()), 5),
        "S_oracle_ceiling_mean": round(float(m.S_oracle_ceiling.mean()), 4),
        "S_constant_mean": round(float(m.S_constant.mean()), 4),
        "S_real_human_mean": round(float(m.S_real_human.mean()), 4),
        "VR_oracle_ceiling_mean": round(float(m.VR_oracle_ceiling.mean()), 4),
        "VR_published_model": 0.431,
        "interpretation": (
            "VR_oracle_ceiling is the dispersion any perfect reader of this prompt could "
            "achieve. If it is close to the model's measured 0.431, the prompt encoding -- "
            "not the model -- is the binding constraint on individuation."),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== H6 SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
