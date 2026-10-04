"""Adapter for the pinned official GEPA 0.1.4 engine.

Only optimization-panel respondent indices enter GEPA. Its per-example metric is
negative raw RPS, which is additive across respondents. The outer ledger is the
forecast budget authority; GEPA's metric-call stopper is advisory.
"""

import inspect
import json
from importlib.metadata import version

import numpy as np

from src.optim import scoring
from src.optim.arms import ops
from src.optim.arms.base import Arm
from src.optim.budget import BudgetExhausted
from src.optim.genotype import KEYS, GenotypeParseError, parse_genotype, to_json


def strict_genotype(text, template):
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict) or set(obj) != set(KEYS):
            raise GenotypeParseError("genotype keys changed")
        for k in ("trait_formulations", "facet_formulations"):
            if not isinstance(obj[k], dict) or set(obj[k]) != set(template[k]):
                raise GenotypeParseError(f"{k} keys changed")
            if any(not isinstance(v, str) or not v.strip() for v in obj[k].values()):
                raise GenotypeParseError(f"{k} has a blank or non-string entry")
        for k in ("role_definition", "critic_formulations"):
            if not isinstance(obj[k], str) or not obj[k].strip():
                raise GenotypeParseError(f"{k} is blank or not a string")
        return parse_genotype(text, template)
    except (ValueError, TypeError, KeyError) as exc:
        raise GenotypeParseError(str(exc)) from exc


class RespondentAdapter:
    """GEPAAdapter duck-typed implementation using the frozen respondent route."""

    propose_new_texts = None

    def __init__(self, arm):
        self.arm = arm
        self.seen = set()

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        arm, fit = self.arm, self.arm.fitness
        if arm.ledger.invalid_proposals >= int(arm.cfg.get("max_invalid_proposals", max(20, 5 * arm.ledger.B))):
            raise BudgetExhausted("official GEPA invalid-proposal attempt limit reached")
        idx = np.asarray(batch, dtype=int)
        if len(idx) == 0 or np.any(idx < 0) or np.any(idx >= fit.n_opt):
            raise ValueError("GEPA batch must contain optimization-panel respondent indices")
        text = candidate.get("persona_json", "")
        first_proposal = text not in self.seen
        if first_proposal:
            self.seen.add(text)
            # Candidate generation belongs to GEPA. LM tokens are counted in the
            # reflection callable; an invalid genome is a separate failed attempt.
            arm.ledger.proposal_attempts += 1
        try:
            g = strict_genotype(text, arm.template)
        except GenotypeParseError:
            arm.ledger.invalid_proposals += 1
            arm._native_record(None, idx, None, parse_ok=False)
            return EvaluationBatch(outputs=[None] * len(idx), scores=[-1e6] * len(idx),
                                   trajectories=[{"error": "invalid persona JSON"} for _ in idx] if capture_traces else None)
        count = len(idx) * len(fit.tgt)
        if not arm.ledger.can_charge(count):
            raise BudgetExhausted("official GEPA requested forecasts beyond outer cap")
        P, _ = fit._beliefs(g, idx)
        losses = scoring.per_respondent(scoring.rps(P, fit.Y[idx]))
        if len(losses) != len(idx) or not np.all(np.isfinite(losses)):
            raise ValueError("nonfinite respondent RPS")
        arm.ledger.charge_forecasts(count, prompt_tokens=fit.last_prompt_tokens)
        arm._native_record(g, idx, losses, parse_ok=True)
        traces = None
        if capture_traces:
            ev = (P * np.arange(1, 6)).sum(-1)
            traces = [{"respondent_index": int(i), "item_errors": [
                {"item": str(item), "predicted_mean": round(float(ev[j, k]), 3),
                 "observed": int(fit.Y[i, k])}
                for k, item in enumerate(fit.tgt)]} for j, i in enumerate(idx)]
        return EvaluationBatch(outputs=[{"raw_rps": float(x)} for x in losses],
                               scores=[-float(x) for x in losses], trajectories=traces)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        rows = []
        for trace, score in zip(eval_batch.trajectories or [], eval_batch.scores):
            rows.append({"Inputs": {"optimization_respondent": trace.get("respondent_index")},
                         "Generated Outputs": {"item_predictions": trace.get("item_errors", [])},
                         "Feedback": f"Raw RPS: {-score:.5f}; improve observed-item forecasts"})
        return {k: rows for k in components_to_update}


class OfficialGEPA(Arm):
    name = "gepa_official"

    def _native_record(self, g, idx, losses, parse_ok):
        cid = f"gepa_c{self._seq:04d}"
        self._seq += 1
        full = parse_ok and len(idx) == self.fitness.n_opt and len(set(idx)) == self.fitness.n_opt
        score = float(np.mean(losses)) if parse_ok else float("inf")
        rec = {"candidate_id": cid, "arm": self.name, "operation": "gepa_native_eval",
               "generation": None, "parent_ids": [], "genotype": {k: g[k] for k in KEYS} if parse_ok else None,
               "parse_ok": parse_ok, "rps_raw_opt": score, "rps_cal_opt": score,
               "batch_respondents": [int(i) for i in idx], "full_opt_panel": bool(full),
               "executed_forecasts": len(idx) * len(self.fitness.tgt) if parse_ok else 0,
               "cumulative_forecasts": self.ledger.executed_forecasts,
               "eval_idx": self.ledger.evaluator_calls - 1 if parse_ok else None}
        self.history.append(rec)
        self.writer.append_history(rec)
        if full and score < self.best_score:
            self.best_genotype, self.best_score, self.best_candidate_id = g, score, cid
            self.writer.write_best(g, rec)

    def _reflection_lm(self, prompt):
        kw = {"temperature": float(self.cfg.get("temperature", 0.7)),
              "max_tokens": int(self.cfg.get("max_tokens", 1500))}
        seed = int(self.rng.integers(0, 2**31 - 1))
        sig = inspect.signature(self.mutator.complete)
        if "seed" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            kw["seed"] = seed
        text, ptok, gtok = self.mutator.complete(ops.SYSTEM, prompt, **kw)
        self.ledger.mutator_calls += 1
        self.ledger.prompt_tokens += int(ptok)
        self.ledger.generated_tokens += int(gtok)
        return text

    def search(self, seed):
        import gepa
        if version("gepa") != "0.1.4":
            raise RuntimeError("official GEPA adapter requires gepa==0.1.4")
        mb = min(int(self.cfg.get("minibatch_respondents", 8)), self.fitness.n_opt)
        adapter = RespondentAdapter(self)
        try:
            result = gepa.optimize(
                seed_candidate={"persona_json": to_json(seed)},
                trainset=list(range(self.fitness.n_opt)), valset=None,
                adapter=adapter, reflection_lm=self._reflection_lm,
                candidate_selection_strategy="pareto", reflection_minibatch_size=mb,
                use_merge=bool(self.cfg.get("use_merge", False)),
                max_metric_calls=self.ledger.B * self.fitness.n_opt,
                cache_evaluation=False, seed=int(self.cfg.get("seed", 0)),
                display_progress_bar=False, raise_on_exception=True)
            self.gepa_total_metric_calls = result.total_metric_calls
            self.ledger.gepa_total_metric_calls = result.total_metric_calls
            selected = strict_genotype(result.best_candidate["persona_json"], self.template)
            return selected
        except BudgetExhausted:
            # The official engine can request a batch after its own stop check.
            # The external cap is authoritative; choose its last fully scored elite.
            self.gepa_total_metric_calls = None
            self.ledger.gepa_total_metric_calls = None
            if self.best_genotype is None:
                raise
            return self.best_genotype
