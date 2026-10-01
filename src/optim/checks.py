import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.optim.orientation import flip_answers
from src.optim.panels import ITEM_COLS, load_corpus

ROOT = Path(__file__).resolve().parents[2]
IDS = list(range(1, 121))


def orientation_regression(min_corr):
    df = load_corpus().set_index("case")
    before, after = [], []
    for f in glob.glob(str(ROOT / "results_experiments/evoprompt_iter2/*/*/cluster_*/after_optimization_test_answers.csv")):
        m = pd.read_csv(f).dropna()
        if m.empty:
            continue
        X = m[ITEM_COLS].to_numpy(float)
        H = df.loc[m["case"], ITEM_COLS].to_numpy(float)
        hm = np.nanmean(H, 0)
        before.append(np.corrcoef(np.nanmean(X, 0), hm)[0, 1])
        after.append(np.corrcoef(np.nanmean(flip_answers(X, IDS), 0), hm)[0, 1])
    b, a = float(np.mean(before)), float(np.mean(after))
    print(f"runs={len(before)} per-item mean corr before={b:.3f} after={a:.3f}")
    return 0 if (a >= min_corr and a > b) else 1


def readout_guard(models_yaml, tier, probe_dir, out):
    """Verify the saved GPU probe receipts; this command performs no inference."""
    models = yaml.safe_load(Path(models_yaml).read_text())
    selected = [m for m in models["models"] if str(m["tier"]) == str(tier)]
    if not selected:
        raise ValueError(f"no models in tier {tier}")
    rows = []
    for model in selected:
        path = Path(probe_dir) / f"{model['slug']}.json"
        row = {"slug": model["slug"], "model": model["hf_id"], "probe": str(path)}
        if not path.is_file():
            row["status"] = "missing"
        else:
            try:
                raw = path.read_bytes()
                receipt = json.loads(raw)
                mass = float(receipt["mass_mean"])
                n = int(receipt["n_requests"])
                row.update(mass_mean=mass, n_requests=n, sha256=hashlib.sha256(raw).hexdigest())
                row["status"] = "pass" if (receipt.get("model") == model["hf_id"]
                                           and np.isfinite(mass) and mass >= float(model.get(
                                               "gate_mass_on_scale", models["defaults"]["gate_mass_on_scale"]))
                                           and n >= 160) else "fail"
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                row.update(status="invalid", error=str(exc))
        rows.append(row)
        print(f"{row['slug']}: {row['status']} mass={row.get('mass_mean', 'n/a')} receipt={path}")
    result = {"tier": str(tier), "kind": "saved_probe_receipt_validation", "models": rows,
              "all_pass": all(r["status"] == "pass" for r in rows)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["all_pass"] else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("orientation_regression")
    o.add_argument("--min-corr", type=float, default=0.60)
    r = sub.add_parser("readout_guard")
    r.add_argument("--models-yaml", default="icml2027/configs/models.yaml")
    r.add_argument("--tier", default="1")
    r.add_argument("--probe-dir", default="icml2027/results/aggregates/probe")
    r.add_argument("--out", default="icml2027/results/aggregates/preflight_readout.json")
    a = ap.parse_args()
    if a.cmd == "orientation_regression":
        sys.exit(orientation_regression(a.min_corr))
    sys.exit(readout_guard(a.models_yaml, a.tier, a.probe_dir, a.out))


if __name__ == "__main__":
    main()
