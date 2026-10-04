from __future__ import annotations

import copy
import inspect
import os
import time
from abc import ABC, abstractmethod

import numpy as np

from src.optim.arms import ops
from src.optim.budget import BudgetExhausted, BudgetLedger
from src.optim.genotype import GenotypeParseError, parse_genotype, to_json


class LLMMutator:
    """OpenAI-compatible chat completion against the shared mutator vLLM server."""

    def __init__(self, model, base_url=None, timeout=900.0):
        from openai import OpenAI
        self.model = model
        self.requested_calls = self.completed_calls = self.failed_calls = 0
        self._client = OpenAI(base_url=base_url or os.environ.get("MUTATOR_BASE_URL", "http://127.0.0.1:8000/v1"),
                              api_key=os.environ.get("LOCAL_LLM_API_KEY", "EMPTY"), timeout=timeout, max_retries=0)

    def complete(self, system, user, temperature=0.7, max_tokens=1200, seed=None):
        self.requested_calls += 1
        try:
            r = self._client.chat.completions.create(
                model=self.model, temperature=temperature, max_tokens=max_tokens, seed=seed,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        except Exception:
            self.failed_calls += 1
            raise
        self.completed_calls += 1
        u = r.usage
        return (r.choices[0].message.content or "", int(getattr(u, "prompt_tokens", 0) or 0),
                int(getattr(u, "completion_tokens", 0) or 0))


class Arm(ABC):
    name = "base"
    WORST_PENALTY = 0.05

    def __init__(self, cfg, rng, mutator, template):
        self.cfg, self.rng, self.mutator, self.template = dict(cfg or {}), rng, mutator, template
        self.fitness = self.ledger = self.writer = None
        self.best_genotype, self.best_score, self.best_candidate_id = None, np.inf, None
        self.worst_score = -np.inf
        self.last_candidate_id = None
        self.last_parse_ok, self.last_mutator_tokens = True, (0, 0)
        self._seq = 0
        self.history: list[dict] = []
        self.last_mutator_seed = None

    @abstractmethod
    def search(self, seed_genotype: dict) -> dict:
        """Propose and evaluate candidates until the budget is exhausted; return the chosen genotype."""

    def run(self, fitness, seed_genotype, ledger: BudgetLedger, writer):
        self.fitness, self.ledger, self.writer = fitness, ledger, writer
        ledger.configure(fitness.forward_passes_per_candidate)
        out = self.search(seed_genotype)
        from src.optim.results import atomic_write_json
        selected = next((h for h in reversed(self.history) if h["parse_ok"] and h.get("full_opt_panel", True) and h["genotype"] ==
                         {k: out[k] for k in ("role_definition", "trait_formulations", "facet_formulations",
                                              "critic_formulations")}), None)
        atomic_write_json(writer.dir / "selected_genotype.json", {"genotype": out,
                          "candidate_id": selected["candidate_id"] if selected else None,
                          "selection": "arm_returned"})
        writer.finish(ledger)
        return out

    # -- the only path to the fitness function -------------------------------------------------
    def _eval(self, genotype, generation, parent_ids, operation, parse_ok=True, mutator_tokens=(0, 0)):
        cid = f"g{generation}_c{self._seq:04d}"
        self._seq += 1
        t0 = time.perf_counter()
        if parse_ok:
            if not self.ledger.can_charge(self.fitness.forward_passes_per_candidate):
                raise BudgetExhausted("insufficient budget for full-panel evaluation")
            res = self.fitness.evaluate(genotype)
            score, raw, s0v, mass, ptok = res.rps_cal_opt, res.rps_raw_opt, res.s0_opt, res.mass, res.prompt_tokens
            alphas, true_cal = list(res.alphas), res.rps_cal_true
            passes = self.fitness.forward_passes_per_candidate
        else:
            score = (self.worst_score if np.isfinite(self.worst_score) else 0.5) + self.WORST_PENALTY
            raw = s0v = mass = true_cal = float("nan")
            ptok, alphas, passes = 0, [], 0
        if parse_ok:
            self.ledger.charge(passes, prompt_tokens=ptok)
        is_best = bool(parse_ok and score < self.best_score)
        if is_best:
            self.best_score, self.best_genotype, self.best_candidate_id = score, copy.deepcopy(genotype), cid
        self.worst_score = max(self.worst_score, score)
        rec = {"eval_idx": self.ledger.evaluator_calls - 1 if parse_ok else None, "generation": generation, "candidate_id": cid,
               "parent_ids": list(parent_ids), "operation": operation, "arm": self.name,
               "genotype": {k: genotype[k] for k in ("role_definition", "trait_formulations",
                                                     "facet_formulations", "critic_formulations")},
               "parse_ok": bool(parse_ok), "rps_cal_opt": float(score), "rps_cal_true_opt": true_cal,
               "alpha": alphas, "rps_raw_opt": raw, "s0_opt": s0v, "mass_on_scale": mass, "prompt_tokens": ptok,
               "mutator_prompt_tokens": mutator_tokens[0], "mutator_generated_tokens": mutator_tokens[1],
               "executed_forecasts": passes, "cumulative_forecasts": self.ledger.executed_forecasts,
               "proposal_attempts": self.ledger.proposal_attempts, "mutator_seed": self.last_mutator_seed,
               "wall_s": round(time.perf_counter() - t0, 3), "is_best_so_far": is_best}
        self.history.append(rec)
        self.writer.append_history(rec)
        if is_best:
            self.writer.write_best(self.best_genotype, rec)
        self.last_candidate_id = cid
        return cid, float(score)

    def _ask(self, system, user, fallback, temperature=None):
        t = float(self.cfg.get("temperature", 0.7)) if temperature is None else temperature
        self.last_mutator_seed = int(self.rng.integers(0, 2**31 - 1))
        try:
            kw = {"temperature": t, "max_tokens": int(self.cfg.get("max_tokens", 1500))}
            sig = inspect.signature(self.mutator.complete)
            if "seed" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                kw["seed"] = self.last_mutator_seed
            text, ptok, gtok = self.mutator.complete(system, user, **kw)
        except Exception as e:  # noqa: BLE001 -- a mutator failure is a failed proposal, charged like a bad parse
            if self.cfg.get('fatal_mutator_errors', False):
                raise
            text, ptok, gtok = f"<mutator error {type(e).__name__}>", 0, 0
        self.last_mutator_tokens = (ptok, gtok)
        try:
            if self.cfg.get("strict_genotype", False):
                from src.optim.arms.gepa_official import strict_genotype
                g = strict_genotype(text, self.template)
            else:
                g = parse_genotype(text, self.template)
            self.last_parse_ok = True
            self.ledger.record_proposal(valid=True, prompt_tokens=ptok, generated_tokens=gtok)
            return g
        except GenotypeParseError:
            self.last_parse_ok = False
            self.ledger.record_proposal(valid=False, prompt_tokens=ptok, generated_tokens=gtok)
            return copy.deepcopy(fallback)

    def _eval_asked(self, genotype, generation, parent_ids, operation):
        return self._eval(genotype, generation, parent_ids, operation,
                          parse_ok=self.last_parse_ok, mutator_tokens=self.last_mutator_tokens)

    def _remaining(self):
        if self.ledger.invalid_proposals >= int(self.cfg.get("max_invalid_proposals", max(20, 5 * self.ledger.B))):
            return 0
        return int(self.ledger.remaining_fraction() >= 1 - 1e-9)

    @staticmethod
    def _gjson(g):
        return to_json(g)

    def _mutate(self, g, gen, parents, op="mutation", temperature=None):
        child = self._ask(ops.SYSTEM, ops.MUTATE.format(genotype=self._gjson(g)), fallback=g, temperature=temperature)
        cid, sc = self._eval_asked(child, gen, parents, op)
        return cid, sc, child, self.last_parse_ok
