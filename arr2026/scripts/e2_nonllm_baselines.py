"""
E2 -- Non-LLM cluster baselines on the paper's exact evaluation protocol.

All three NeurIPS reviewers asked for these and none were run. They need no
model at all, so they cost nothing and they bound what any LLM persona has to
beat to be interesting:

  global_mean        answer the grand mean of every item (floor)
  cluster_majority   per-cluster, per-item modal human answer      (FUHt, FHNv, 3bHJ)
  cluster_mean       per-cluster, per-item rounded mean answer
  empirical_sample   sample per-item from the cluster's response distribution (FHNv)
  centroid_determin  map the cluster centroid facet percentile onto a Likert answer (FHNv)
  train_user_copy    copy a random *train* user's answers (a same-cluster human)

Protocol is copied from src/simulator/person_type_opt.py so the numbers are
directly comparable to the paper's:
    first `num_participants` rows of the cluster, in file order
    first 60% -> train (baselines are fitted here), last 40% -> test

Metric is `similarity`, matching src/utils/personality_match.py:
    mean over items of 1 - |model - human| / 4

Also reports the profile-level metrics the paper shows moving the *wrong* way,
so E2 speaks directly to the answer-vs-profile tension.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ITEMS = [f"i{i}" for i in range(1, 121)]
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
FACETS = [
    "facet_imagination", "facet_artistic_interests", "facet_emotionality",
    "facet_adventurousness", "facet_intellect", "facet_liberalism",
    "facet_self_efficacy", "facet_orderliness", "facet_dutifulness",
    "facet_achievement_striving", "facet_self_discipline", "facet_cautiousness",
    "facet_friendliness", "facet_gregariousness", "facet_assertiveness",
    "facet_activity_level", "facet_excitement_seeking", "facet_cheerfulness",
    "facet_trust", "facet_morality", "facet_altruism", "facet_cooperation",
    "facet_modesty", "facet_sympathy",
    "facet_anxiety", "facet_anger", "facet_depression",
    "facet_self_consciousness", "facet_immoderation", "facet_vulnerability",
]


def similarity(pred, human):
    """Mean answer-level similarity on the 1..5 Likert scale, as in personality_match."""
    return float(np.mean(1.0 - np.abs(np.asarray(pred, float) - np.asarray(human, float)) / 4.0))


def bootstrap_ci(vals, n_boot=5000, seed=2026):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    means = [rng.choice(v, len(v), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/df_ipipneo_120_clusters")
    ap.add_argument("--out", default="arr2026/results/e2")
    ap.add_argument("--num-participants", type=int, default=100)
    ap.add_argument("--clusters", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--extra-splits", type=int, default=20,
                    help="additional random splits, to show how much the fixed "
                         "first-100-rows protocol moves (feeds E5)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    rng = np.random.default_rng(args.seed)

    per_user, per_cell = [], []

    def run_split(cluster, train, test, split_id):
        tr_items = train[ITEMS].to_numpy(float)
        te_items = test[ITEMS].to_numpy(float)

        majority = np.array([pd.Series(tr_items[:, j]).mode().iloc[0] for j in range(120)])
        meanans = np.round(tr_items.mean(axis=0))
        grand = np.round(np.full(120, df[ITEMS].to_numpy(float).mean()))
        # centroid facet percentile (1..99) -> Likert 1..5
        centroid_pct = train[FACETS].to_numpy(float).mean(axis=0).mean()
        centroid_ans = np.full(120, np.clip(round(1 + 4 * (centroid_pct - 1) / 98.0), 1, 5))

        preds = {
            "global_mean": lambda i: grand,
            "cluster_majority": lambda i: majority,
            "cluster_mean": lambda i: meanans,
            "centroid_determin": lambda i: centroid_ans,
            "empirical_sample": lambda i: np.array(
                [rng.choice(tr_items[:, j]) for j in range(120)]),
            "train_user_copy": lambda i: tr_items[rng.integers(len(tr_items))],
        }

        for name, fn in preds.items():
            sims = []
            for i in range(len(te_items)):
                s = similarity(fn(i), te_items[i])
                sims.append(s)
                if split_id == "paper":
                    per_user.append({"cluster": cluster, "baseline": name,
                                     "case": test.iloc[i].get("case"), "similarity": s})
            lo, hi = bootstrap_ci(sims, seed=args.seed)
            per_cell.append({
                "split": split_id, "cluster": cluster, "baseline": name,
                "n_test": len(sims),
                "similarity_mean": float(np.mean(sims)),
                "similarity_std": float(np.std(sims)),
                "ci95_lo": lo, "ci95_hi": hi,
            })

    for c in args.clusters:
        sub = df[df["clusters"] == c]
        # --- the paper's exact protocol ---
        total = sub.iloc[: args.num_participants]
        tsize = int(args.num_participants * 0.6)
        run_split(c, total.iloc[:tsize], total.iloc[tsize:], "paper")
        print(f"cluster {c}: paper split done", flush=True)

        # --- random splits, same sizes, to expose protocol variance ---
        for s in range(args.extra_splits):
            r = np.random.default_rng(args.seed + 1000 * c + s)
            pick = sub.iloc[r.choice(len(sub), args.num_participants, replace=False)]
            run_split(c, pick.iloc[:tsize], pick.iloc[tsize:], f"rand{s}")
        print(f"cluster {c}: {args.extra_splits} random splits done", flush=True)

    cells = pd.DataFrame(per_cell)
    cells.to_csv(out / "baseline_cells.csv", index=False)
    pd.DataFrame(per_user).to_csv(out / "baseline_per_user.csv", index=False)

    paper = cells[cells.split == "paper"]
    rand = cells[cells.split != "paper"]
    summary = {
        "paper_protocol": paper.groupby("baseline")
            .similarity_mean.mean().sort_values(ascending=False).round(4).to_dict(),
        "paper_protocol_by_cluster": {
            str(c): paper[paper.cluster == c].set_index("baseline")
                .similarity_mean.round(4).to_dict() for c in args.clusters},
        "random_splits_mean": rand.groupby("baseline")
            .similarity_mean.mean().round(4).to_dict(),
        "random_splits_std_across_splits": rand.groupby("baseline")
            .similarity_mean.std().round(4).to_dict(),
        "note": "Compare `cluster_majority` against the paper's reported LLM "
                "answer-similarity. If it is competitive, the evolutionary "
                "prompt result needs reframing.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== E2 SUMMARY ===")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
