"""The only code that reads the evaluation panel. Called once per cell."""
import hashlib

import numpy as np

from src.optim import scoring as S
from src.optim.orientation import flip_simplex
from src.optim.panels import ITEM_SPLIT, answers, persona_scores
from src.optim.persona import system_prompt


def paired_boot_ci(a, b, B=2000, seed=20260925):
    d = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(seed)
    n = len(d)
    bs = np.array([np.nanmean(d[rng.integers(0, n, n)]) for _ in range(B)])
    return float(np.nanmean(d)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def beliefs_for(readout, g, df, tgt):
    systems = [system_prompt(g, persona_scores(x)) for x in answers(df, ITEM_SPLIT[0])]
    P, M, _ = readout.beliefs(systems, tgt)
    return flip_simplex(P, tgt), M


def _score_arm(readout, g, panels, F0, alphas):
    tgt = ITEM_SPLIT[1]
    Ycal, Yev = answers(panels.cal, tgt), answers(panels.eval, tgt)
    Pc, _ = beliefs_for(readout, g, panels.cal, tgt)
    a, _ = S.select_alpha(S.cdf(Pc), F0, Ycal, alphas)
    Pe, Me = beliefs_for(readout, g, panels.eval, tgt)
    per = S.per_respondent(S.cal_cells(S.cdf(Pe), F0, Yev, a))
    mu = (Pe * np.arange(1, 6)).sum(-1)
    vr = float(np.var(mu, 0).mean() / np.nanvar(Yev, 0).mean())
    return ({"rps_cal": float(np.nanmean(per)), "rps_raw": float(np.nanmean(S.rps(Pe, Yev))),
             "s0": float(np.nanmean(S.s0(Pe, Yev))), "alpha": a, "vr_between": vr,
             "mass_on_scale": float(Me.mean())}, per, Pe, Pc, a)


def floors(panels, draws=20, seed=0):
    tgt = ITEM_SPLIT[1]
    Ytr, Yev = answers(panels.train, tgt), answers(panels.eval, tgt)
    n, J = Yev.shape

    def point(v):
        P = np.zeros((n, J, 5))
        P[np.arange(n)[:, None], np.arange(J)[None, :], (np.clip(np.rint(v), 1, 5) - 1).astype(int)] = 1
        return P

    F0 = S.pop_cdf(Ytr)
    rng = np.random.default_rng(seed)
    human = np.mean([np.nanmean(S.rps(point(np.broadcast_to(Ytr[rng.integers(len(Ytr))], (n, J))), Yev))
                     for _ in range(draws)])
    return {"cluster_mean_rps": float(np.nanmean(S.rps(point(np.broadcast_to(np.nanmean(Ytr, 0), (n, J))), Yev))),
            "cluster_prior_rps": float(np.nanmean(S.rps(S.cdf_to_simplex(np.broadcast_to(F0, (n, J, 4))), Yev))),
            "human_rps": float(human)}


def evaluate_frozen(panels, readout, base_g, evolved_g, alphas, boot_B=2000, boot_seed=20260925,
                    wrong_persona_seed=0):
    tgt = ITEM_SPLIT[1]
    F0 = S.pop_cdf(answers(panels.train, tgt))
    Yev = answers(panels.eval, tgt)
    base, pb, Pb, _, _ = _score_arm(readout, base_g, panels, F0, alphas)
    evo, pe, Pe, _, a_e = _score_arm(readout, evolved_g, panels, F0, alphas)
    # wrong-persona control: evolved genotype, personas permuted within the eval panel, alpha from cal
    perm = np.random.default_rng(wrong_persona_seed).permutation(len(panels.eval))
    Pw, _ = beliefs_for(readout, evolved_g, panels.eval.iloc[perm], tgt)
    evo["wrong_persona_rps_cal"] = float(np.nanmean(S.cal_cells(S.cdf(Pw), F0, Yev, a_e)))
    m, lo, hi = paired_boot_ci(pb, pe, boot_B, boot_seed)
    ids = ",".join(map(str, sorted(panels.eval["case"].tolist())))
    excl = evo["mass_on_scale"] < 0.5 or base["mass_on_scale"] < 0.5
    return {"panel": {"cluster": int(panels.cluster), "seed": int(panels.seed), "n_eval": len(panels.eval),
                      "respondent_ids_sha256": hashlib.sha256(ids.encode()).hexdigest()},
            "base": base, "evolved": evo,
            "delta": {"rps_cal": m, "rps_cal_ci95": [lo, hi], "boot_B": boot_B, "boot_seed": boot_seed},
            "floors": floors(panels), "excluded": bool(excl),
            "exclusion_reason": "mass_on_scale<0.5" if excl else None,
            "per_respondent": {"base_rps_cal": pb.tolist(), "evolved_rps_cal": pe.tolist()},
            "_beliefs": {"base": Pb.astype(np.float32), "evolved": Pe.astype(np.float32),
                         "wrong": Pw.astype(np.float32)}}
