"""Search-time fitness: calibrated RPS on the optimisation panel, alpha cross-fitted inside it.
Never reads the calibration or evaluation panels."""
import time
from dataclasses import dataclass

import numpy as np

from src.optim import scoring as S
from src.optim.genotype import to_json
from src.optim.orientation import flip_simplex
from src.optim.panels import ITEM_SPLIT, TEXT_OF_ITEM, answers, persona_scores
from src.optim.persona import system_prompt


@dataclass
class FitnessResult:
    rps_cal_opt: float
    rps_raw_opt: float
    s0_opt: float
    mass: float
    prompt_tokens: int
    wall_s: float
    alphas: tuple
    rps_cal_true: float = float("nan")


class Fitness:
    def __init__(self, panels, readout, alphas, rng, objective="rps_cal"):
        self.readout, self.alphas, self.rng, self.objective = readout, list(alphas), rng, objective
        inp, self.tgt = ITEM_SPLIT
        self.F0 = S.pop_cdf(answers(panels.train, self.tgt))
        self.Y = answers(panels.opt, self.tgt)
        self.scores = [persona_scores(x) for x in answers(panels.opt, inp)]
        self.human_between_var = float(np.nanvar(self.Y, axis=0).mean())
        self.n_opt = len(self.Y)
        self.forward_passes_per_candidate = self.n_opt * len(self.tgt)
        self.last_prompt_tokens = 0
        self._cache = {}
        self._summ = {}      # key -> small, never-evicted summaries used by feedback() and per_respondent_scores()
        self._last_key = None

    def _beliefs(self, g, idx):
        systems = [system_prompt(g, self.scores[i]) for i in idx]
        P, M, tok = self.readout.beliefs(systems, self.tgt)
        self.last_prompt_tokens = tok
        return flip_simplex(P, self.tgt), M

    def _result(self, P, M, Y, t0):
        cal, alphas = S.crossfit_rps_cal(S.cdf(P), self.F0, Y, self.alphas, self.rng)
        raw = float(np.nanmean(S.rps(P, Y)))
        s0v = float(np.nanmean(S.s0(P, Y)))
        primary = s0v if self.objective == "s0" else cal
        return FitnessResult(float(primary), raw, s0v, float(M.mean()), self.last_prompt_tokens,
                             time.perf_counter() - t0, tuple(alphas), float(cal)), cal

    def evaluate(self, g) -> FitnessResult:
        t0 = time.perf_counter()
        P, M = self._beliefs(g, range(self.n_opt))
        res, cal = self._result(P, M, self.Y, t0)
        key = to_json(g)
        self._last_key = key
        a = float(np.mean(res.alphas))
        per = S.per_respondent(S.cal_cells(S.cdf(P), self.F0, self.Y, a))
        if self.objective == "s0":
            per = S.per_respondent(S.s0(P, self.Y))
        self._put(key, {"P": P, "per": per, "cal": cal})
        self._summ[key] = self._summary(P, self.Y, per)
        return res

    @staticmethod
    def _summary(P, Y, per=None):
        ev = (P * np.arange(1, 6)).sum(-1)
        return {"err": np.nanmean(ev - Y, axis=0), "pred": np.nanmean(ev, axis=0), "hum": np.nanmean(Y, axis=0),
                "per": per}

    def _put(self, key, val):
        """LRU insert: an existing key is moved to the end, so the entry just written is never the one evicted
        (re-evaluating an identical genotype used to leave it in its old slot, where eviction removed it)."""
        self._cache.pop(key, None)
        while len(self._cache) >= 32:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = val

    def evaluate_subset(self, g, idx) -> FitnessResult:
        t0 = time.perf_counter()
        idx = np.asarray(idx)
        P, M = self._beliefs(g, idx)
        key = to_json(g)
        self._put("sub:" + key, {"P": P, "idx": idx})
        self._summ["sub:" + key] = self._summary(P, self.Y[idx])
        return self._result(P, M, self.Y[idx], t0)[0]

    def per_respondent_scores(self, g):
        return self._summ[to_json(g)]["per"]

    def feedback(self, g, k=10, prefer_subset=False) -> str:
        key = to_json(g)
        sk = "sub:" + key
        c = self._summ[sk] if (prefer_subset and sk in self._summ) or key not in self._summ else self._summ[key]
        err, pred, hum = c["err"], c["pred"], c["hum"]
        order = np.argsort(-np.abs(err))[:k]
        return "\n".join(
            f"item {self.tgt[j]} '{TEXT_OF_ITEM[self.tgt[j]]}' (scored so that higher = more of the trait): "
            f"model too {'high' if err[j] > 0 else 'low'} by {abs(err[j]):.2f} "
            f"(mean predicted {pred[j]:.2f}, human {hum[j]:.2f})"
            for j in order)

    def last_descriptor(self):
        c = self._cache[self._last_key]
        P = c["P"]
        mu = (P * np.arange(1, 6)).sum(-1)
        sd = np.sqrt((P * (np.arange(1, 6) - mu[..., None]) ** 2).sum(-1)).mean()
        return {"prompt_length_chars": len(self._last_key), "predicted_sd": float(sd)}

    def last_objectives(self):
        c = self._cache[self._last_key]
        mu = (c["P"] * np.arange(1, 6)).sum(-1)
        vr = float(np.var(mu, axis=0).mean() / self.human_between_var)
        return {"rps_cal": float(c["cal"]), "neg_vr_between": -vr}
