"""
Pilot for idea 3 -- two-knob belief calibration.

Question: is the missing individuation RECOVERABLE from the belief distributions
we already have, or is the persona signal simply not there?

On the saved five-way probability vectors we re-softmax with two independent knobs:

    q_ij = softmax( ( Lbar_j + alpha * (L_ij - Lbar_j) ) / T )

where L_ij = log p_ij, Lbar_j is the per-item mean logit across personas.
  alpha  amplifies the PERSONA-SPECIFIC residual  -> should raise VR_between
  T      scales overall sharpness                 -> should raise VR_within

If a setting exists that lifts VR_between toward human without wrecking the
proper score (CRPS) or the per-item marginal (W1), the persona signal is present
but attenuated, and post-hoc calibration repairs it. If VR_between only rises by
degrading CRPS/W1 one-for-one, the signal is misdirected or absent -- which kills
calibration as a repair but sharpens the mechanism claim.

Split: alpha,T chosen on 24 facets, reported on 6 held-out facets. No GPU.
"""
import json, pathlib
import numpy as np, pandas as pd

ITEMS = [f"i{i}" for i in range(1, 121)]
ROOT = pathlib.Path(".")
VALS = np.arange(1, 6, dtype=float)
KEY = pd.read_csv("data/IPIP-NEO/120/item_key.csv").sort_values("item")
FACET = KEY["facet_key"].to_numpy()
df = pd.read_csv("data/raw/df_ipipneo_120_clusters")

MODELS = [("qwen25_7b","hse"),("qwen25_32b","hse"),("qwen25_72b","hse"),
          ("mistral_24b","hse"),("qwq_32b","hse"),("qwen3_235b","euler")]
ALPHAS = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
TEMPS  = [1.0, 1.5, 2.0, 3.0]


def metrics(P, human_items, hs, cols):
    """VR_within / VR_between / CRPS / W1 on a subset of item columns."""
    ok = hs[cols] > 0
    mu  = np.nansum(P[:, cols, :] * VALS, axis=2)
    ex2 = np.nansum(P[:, cols, :] * VALS**2, axis=2)
    var = np.clip(ex2 - mu**2, 0, None)
    wv, bv = np.nanmean(var, axis=0), np.nanvar(mu, axis=0)
    vr_w = np.nanmean(np.sqrt(wv[ok]) / hs[cols][ok])
    vr_b = np.nanmean(np.sqrt(bv[ok]) / hs[cols][ok])
    # CRPS of the mixture against the human marginal, and W1, per item
    crps, w1 = [], []
    for k, j in enumerate(cols):
        y = human_items[:, j]; y = y[~np.isnan(y)]
        if len(y) == 0: continue
        pm = np.nanmean(P[:, j, :], axis=0)          # persona-mixture pmf
        F = np.cumsum(pm)
        Fy = np.array([(y <= v).mean() for v in VALS])
        crps.append(np.sum((F - Fy) ** 2))            # ordinal CRPS
        w1.append(np.sum(np.abs(F - Fy)))
    return vr_w, vr_b, float(np.mean(crps)), float(np.mean(w1))


rows = []
for tag, src in MODELS:
    Ps, hss = [], []
    for cl in range(4):
        f = ROOT / f"arr2026/results_{src}/h4_{tag}/readout_cluster_{cl}/belief_probs.npy"
        if not f.exists(): continue
        Ps.append(np.load(f))
        hss.append((np.nanstd(df[df.clusters == cl][ITEMS].to_numpy(float), axis=0),
                    df[df.clusters == cl][ITEMS].to_numpy(float)))
    if not Ps: continue

    facets = sorted(set(FACET))
    dev = np.array([j for j in range(120) if FACET[j] in facets[:24]])
    hld = np.array([j for j in range(120) if FACET[j] in facets[24:]])

    best, best_score = None, -1e9
    for a in ALPHAS:
        for T in TEMPS:
            vw = vb = cr = 0.0
            for P, (hs, hi) in zip(Ps, hss):
                L = np.log(np.clip(P, 1e-12, None))
                Lb = L.mean(axis=0, keepdims=True)
                Q = np.exp((Lb + a * (L - Lb)) / T)
                Q /= Q.sum(axis=2, keepdims=True)
                w, b, c, _ = metrics(Q, hi, hs, dev)
                vw += w; vb += b; cr += c
            n = len(Ps); vw, vb, cr = vw/n, vb/n, cr/n
            # objective: maximise individuation subject to not degrading CRPS
            score = vb - 2.0 * max(0.0, cr - base_crps) if (base_crps := None) else vb
            if a == 1.0 and T == 1.0:
                base = (vw, vb, cr)
            if score > best_score:
                best_score, best = score, (a, T, vw, vb, cr)

    a, T, _, _, _ = best
    vw = vb = cr = w1 = 0.0
    bw = bb = bc = bw1 = 0.0
    for P, (hs, hi) in zip(Ps, hss):
        L = np.log(np.clip(P, 1e-12, None)); Lb = L.mean(axis=0, keepdims=True)
        Q = np.exp((Lb + a * (L - Lb)) / T); Q /= Q.sum(axis=2, keepdims=True)
        w, b, c, ww = metrics(Q, hi, hs, hld); vw += w; vb += b; cr += c; w1 += ww
        w, b, c, ww = metrics(P, hi, hs, hld); bw += w; bb += b; bc += c; bw1 += ww
    n = len(Ps)
    rows.append({"model": tag, "alpha": a, "T": T,
                 "base_VR_within": bw/n, "base_VR_between": bb/n,
                 "base_CRPS": bc/n, "base_W1": bw1/n,
                 "cal_VR_within": vw/n, "cal_VR_between": vb/n,
                 "cal_CRPS": cr/n, "cal_W1": w1/n})
    print(f"{tag:<13} a={a:<4} T={T:<4} | between {bb/n:.3f} -> {vb/n:.3f} | "
          f"CRPS {bc/n:.4f} -> {cr/n:.4f} | W1 {bw1/n:.3f} -> {w1/n:.3f}", flush=True)

t = pd.DataFrame(rows)
out = ROOT / "idea-stage"; out.mkdir(exist_ok=True)
t.to_csv(out / "pilot_two_knob_results.csv", index=False)
d_bet = (t.cal_VR_between - t.base_VR_between).mean()
d_crps = (t.cal_CRPS - t.base_CRPS).mean()
summary = {"n_models": len(t),
           "mean_delta_VR_between": round(float(d_bet), 4),
           "mean_delta_CRPS": round(float(d_crps), 5),
           "mean_cal_VR_between": round(float(t.cal_VR_between.mean()), 4),
           "human_reference": 1.0,
           "verdict": ("RECOVERABLE: individuation rises without degrading the proper score"
                       if d_bet > 0.03 and d_crps <= 0.002 else
                       "TRADED: individuation only rises by degrading CRPS"
                       if d_bet > 0.03 else
                       "NOT RECOVERABLE: calibration cannot raise individuation")}
(out / "pilot_two_knob_summary.json").write_text(json.dumps(summary, indent=2))
print("\n=== PILOT SUMMARY ===")
print(json.dumps(summary, indent=2))
