# Optimizer Arms Implementation Plan (icml2027, plan 2 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement ten budget-matched prompt-optimizer arms behind one interface so a cell runner can call `ARMS[name](cfg, rng, mutator, template).run(fitness, seed_genotype, ledger, writer)` and get back the best genotype selected on the optimisation panel.

**Architecture:** Every arm is a subclass of `Arm` in `src/optim/arms/base.py`. The base class owns the only path to the fitness function (`_eval`), which charges the `BudgetLedger`, appends to `history.jsonl`, and tracks the best-so-far genotype. Arms differ only in how they propose candidates. All LLM calls go through one `Mutator` with the prompt templates in `src/optim/arms/ops.py`, so the mutator model is identical across arms.

**Tech Stack:** Python 3.11, numpy, the `openai` client against a local vLLM server (mutator), pytest. Consumes plan 1's `Fitness`, `FitnessResult`, `BudgetLedger`, `CellWriter`, `parse_genotype`, `GenotypeParseError`.

**Spec:** `icml2027/PLAN.md` §4 E1, E3, E5; `icml2027/PREREGISTRATION.md` §2 (budget B = 80); `icml2027/configs/optimizers/*.yaml`.

## Global Constraints

- Budget B = 80 candidate evaluations per cell; every arm must exhaust exactly B (the base class raises `BudgetExhausted`; arms stop cleanly when `ledger.remaining == 0`).
- A candidate that fails to parse is still charged and recorded with `fitness = worst_so_far + 0.05` (RPS is lower-is-better); it never enters the population.
- The mutator model is fixed per run (`configs/models.yaml: mutator`), `temperature 0.7`, `max_tokens 1200`.
- No arm reads the calibration or evaluation panel. Only `Fitness.evaluate` is called.
- Every random choice uses the `numpy.random.Generator` passed in; no `random` module.
- Files stay small: one arm per file, ≤ 150 lines.

---

### Task 1: Arm base class, mutator, ops templates

**Files:**
- Create: `src/optim/arms/base.py`
- Create: `src/optim/arms/ops.py`
- Create: `src/optim/arms/__init__.py` (registry)
- Test: `tests/optim/test_arms_base.py`

**Interfaces:**
- Consumes (plan 1): `Fitness.evaluate(genotype: dict) -> FitnessResult` with fields `rps_cal_opt: float, rps_raw_opt: float, s0_opt: float, mass: float, prompt_tokens: int, wall_s: float, alphas: tuple`; `BudgetLedger(B).charge(passes, prompt_tokens=0, generated_tokens=0)`, `.remaining`, `BudgetExhausted`; `CellWriter.append_history(record: dict)`, `.write_best(genotype: dict, record: dict)`; `parse_genotype(text: str, template: dict) -> dict`, `GenotypeParseError`.
- Produces: `class Mutator(Protocol): complete(system: str, user: str, temperature: float, max_tokens: int) -> tuple[str, int, int]`; `class LLMMutator(Mutator)`; `class Arm(ABC)` with `run(fitness, seed_genotype, ledger, writer) -> dict`, `_eval(genotype, generation, parent_ids, operation) -> tuple[str, float]` (candidate_id, fitness), `_ask(system, user, fallback) -> dict`; `ARMS: dict[str, type[Arm]]`; `ops.MUTATE, ops.CROSSOVER, ops.PARAPHRASE, ops.OPRO, ops.REFLECT, ops.DE_STEP, ops.META_MUTATE, ops.GENOTYPE_SCHEMA`.

- [ ] **Step 1: Write the failing test**

```python
# tests/optim/test_arms_base.py
import numpy as np
from src.optim.arms.base import Arm, Mutator
from src.optim.budget import BudgetLedger
from tests.optim.conftest import FakeFitness, FakeWriter, TEMPLATE

class EchoMutator:
    def complete(self, system, user, temperature=0.7, max_tokens=1200):
        # returns the template genotype with the role suffixed, as JSON
        import json, copy
        g = copy.deepcopy(TEMPLATE); g["role_definition"] += " (mutated)"
        return json.dumps(g), 100, 50

class TwoStepArm(Arm):
    name = "twostep"
    def run(self, fitness, seed_genotype, ledger, writer):
        self._eval(seed_genotype, 0, [], "init")
        g = self._ask("sys", "user", fallback=seed_genotype)
        self._eval(g, 1, [self.last_candidate_id], "mutation")
        return self.best_genotype

def test_eval_charges_budget_and_tracks_best():
    fit, writer = FakeFitness(), FakeWriter()
    arm = TwoStepArm(cfg={}, rng=np.random.default_rng(0), mutator=EchoMutator(), template=TEMPLATE)
    ledger = BudgetLedger(B=80)
    best = arm.run(fit, TEMPLATE, ledger, writer)
    assert ledger.used == 2
    assert len(writer.history) == 2
    assert writer.history[1]["parent_ids"] == [writer.history[0]["candidate_id"]]
    assert best["role_definition"].endswith("(mutated)") or best == TEMPLATE
    assert writer.best is not None

def test_unparsable_is_charged_and_scored_worst():
    class BadMutator:
        def complete(self, *a, **k): return "not json at all", 10, 10
    fit, writer = FakeFitness(), FakeWriter()
    arm = TwoStepArm(cfg={}, rng=np.random.default_rng(0), mutator=BadMutator(), template=TEMPLATE)
    ledger = BudgetLedger(B=80)
    arm.run(fit, TEMPLATE, ledger, writer)
    assert ledger.used == 2
    assert writer.history[1]["parse_ok"] is False
    assert writer.history[1]["rps_cal_opt"] > writer.history[0]["rps_cal_opt"]
```

`tests/optim/conftest.py` (created in plan 1) must export `FakeFitness` (returns `FitnessResult` with `rps_cal_opt = 0.20 - 0.001 * len(genotype["role_definition"])`, clipped to `[0.05, 0.5]`), `FakeWriter` (collects `history: list`, `best`), and `TEMPLATE` (a valid genotype dict).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/optim/test_arms_base.py -q`
Expected: FAIL with `ModuleNotFoundError: src.optim.arms.base`

- [ ] **Step 3: Write ops templates**

```python
# src/optim/arms/ops.py
"""Prompt templates for every LLM operation an arm can request. One place, so the
mutator sees identical instructions regardless of which arm asks."""

GENOTYPE_SCHEMA = """Return ONLY a JSON object with exactly these keys:
{"role_definition": "<one paragraph: who the respondent is>",
 "trait_formulations": {"<trait>": "<one sentence describing this Big Five trait>", ...},
 "facet_formulations": {"<facet>": "<one sentence describing this facet>", ...},
 "critic_formulations": "<one paragraph: how to stay in character when answering>"}
Keep every key present in the input. Do not add keys. Do not add commentary."""

SYSTEM = ("You improve persona descriptions used to make a language model answer a "
          "personality questionnaire the way one specific person would. The persona is a "
          "JSON object. Intensity words (e.g. 'slightly', 'very strongly') are inserted "
          "automatically per person; you only edit the wording of the descriptions.\n" + GENOTYPE_SCHEMA)

MUTATE = ("Rewrite this persona so it makes the answering model more sensitive to the "
          "person's actual trait levels while staying fluent. Change wording, emphasis or "
          "structure of at least two entries.\n\nPERSONA:\n{genotype}")

CROSSOVER = ("Combine the two personas below into ONE child persona: keep the more precise "
             "formulation of each trait and facet, and write a role_definition that merges both.\n\n"
             "PARENT A:\n{a}\n\nPARENT B:\n{b}")

PARAPHRASE = ("Paraphrase every description in this persona, preserving meaning exactly. Do not "
              "add or remove information.\n\nPERSONA:\n{genotype}")

OPRO = ("Below are persona candidates with their scores (LOWER score is better; it is a "
        "probabilistic error on held-out questionnaire answers). Study what the better ones do "
        "differently and propose ONE new persona you expect to score lower than all of them.\n\n"
        "{history}")

REFLECT = ("This persona was used to predict a real person's answers. Below is diagnostic "
           "feedback: the items where predictions were worst, with the direction of error "
           "('model too high' means the model over-endorsed the item). Reflect on what the "
           "persona wording caused, then return an improved persona.\n\nPERSONA:\n{genotype}\n\n"
           "FEEDBACK:\n{feedback}")

DE_STEP = ("Differential evolution step. Identify what differs between persona A and persona B "
           "(the 'difference vector'), apply that same kind of change to persona C, then "
           "cross the result with persona X by keeping X's entry wherever the changed entry is "
           "not clearly better. Return the final persona.\n\nA:\n{a}\n\nB:\n{b}\n\nC:\n{c}\n\nX:\n{x}")

META_MUTATE = ("You write instructions that other editors follow to improve personas. Here is a "
               "current instruction:\n\n{mutation_prompt}\n\nWrite a different instruction of "
               "similar length that would lead to more useful edits. Return ONLY the instruction text.")
```

- [ ] **Step 4: Write the base class and mutator**

```python
# src/optim/arms/base.py
from __future__ import annotations
import copy, json, os, time
from abc import ABC, abstractmethod
from typing import Protocol
import numpy as np
from src.optim.budget import BudgetLedger, BudgetExhausted
from src.optim.genotype import parse_genotype, GenotypeParseError, to_json
from src.optim.arms import ops

class Mutator(Protocol):
    def complete(self, system: str, user: str, temperature: float = 0.7,
                 max_tokens: int = 1200) -> tuple[str, int, int]: ...

class LLMMutator:
    """OpenAI-compatible chat completion against a local vLLM server (or any provider URL)."""
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 timeout: float = 600.0):
        from openai import OpenAI
        self.model = model
        self._client = OpenAI(base_url=base_url or os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
                              api_key=api_key or os.environ.get("LOCAL_LLM_API_KEY", "EMPTY"), timeout=timeout)
    def complete(self, system, user, temperature=0.7, max_tokens=1200):
        r = self._client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        u = r.usage
        return r.choices[0].message.content or "", int(getattr(u, "prompt_tokens", 0) or 0), int(getattr(u, "completion_tokens", 0) or 0)

class Arm(ABC):
    name: str = "base"
    WORST_PENALTY = 0.05

    def __init__(self, cfg: dict, rng: np.random.Generator, mutator: Mutator, template: dict):
        self.cfg, self.rng, self.mutator, self.template = cfg, rng, mutator, template
        self.fitness = None; self.ledger = None; self.writer = None
        self.best_genotype = None; self.best_score = np.inf; self.best_candidate_id = None
        self.worst_score = -np.inf; self.last_candidate_id = None; self._seq = 0
        self.history: list[dict] = []

    @abstractmethod
    def run(self, fitness, seed_genotype: dict, ledger: BudgetLedger, writer) -> dict: ...

    def _bind(self, fitness, ledger, writer):
        self.fitness, self.ledger, self.writer = fitness, ledger, writer

    def _eval(self, genotype: dict, generation: int, parent_ids: list[str], operation: str,
              parse_ok: bool = True, mutator_tokens: tuple[int, int] = (0, 0)) -> tuple[str, float]:
        if self.fitness is None: raise RuntimeError("call _bind first")
        cid = f"g{generation}_c{self._seq:04d}"; self._seq += 1
        t0 = time.perf_counter()
        if parse_ok:
            res = self.fitness.evaluate(genotype)
            score, raw, s0v, mass, ptok, alphas = (res.rps_cal_opt, res.rps_raw_opt, res.s0_opt,
                                                    res.mass, res.prompt_tokens, list(res.alphas))
            passes = self.fitness.forward_passes_per_candidate
        else:
            score = (self.worst_score if np.isfinite(self.worst_score) else 0.5) + self.WORST_PENALTY
            raw, s0v, mass, ptok, alphas, passes = float("nan"), float("nan"), float("nan"), 0, [], 0
        self.ledger.charge(passes, prompt_tokens=ptok + mutator_tokens[0], generated_tokens=mutator_tokens[1])
        is_best = parse_ok and score < self.best_score
        if is_best:
            self.best_score, self.best_genotype, self.best_candidate_id = score, copy.deepcopy(genotype), cid
        self.worst_score = max(self.worst_score, score)
        rec = {"eval_idx": self.ledger.used - 1, "generation": generation, "candidate_id": cid,
               "parent_ids": list(parent_ids), "operation": operation, "arm": self.name,
               "genotype": genotype, "parse_ok": parse_ok, "rps_cal_opt": float(score),
               "alpha": alphas, "rps_raw_opt": raw, "s0_opt": s0v, "mass_on_scale": mass,
               "prompt_tokens": ptok, "mutator_prompt_tokens": mutator_tokens[0],
               "mutator_generated_tokens": mutator_tokens[1],
               "wall_s": round(time.perf_counter() - t0, 3), "is_best_so_far": bool(is_best)}
        self.history.append(rec); self.writer.append_history(rec)
        if is_best: self.writer.write_best(self.best_genotype, rec)
        self.last_candidate_id = cid
        return cid, float(score)

    def _ask(self, system: str, user: str, fallback: dict) -> dict:
        """Ask the mutator for a genotype. Returns (genotype, ok, tokens) via attributes:
        on parse failure returns a copy of `fallback` and sets self.last_parse_ok=False."""
        text, ptok, gtok = self.mutator.complete(system, user, temperature=float(self.cfg.get("temperature", 0.7)),
                                                 max_tokens=int(self.cfg.get("max_tokens", 1200)))
        self.last_mutator_tokens = (ptok, gtok)
        try:
            g = parse_genotype(text, self.template); self.last_parse_ok = True; return g
        except GenotypeParseError:
            self.last_parse_ok = False; return copy.deepcopy(fallback)

    def _eval_asked(self, genotype: dict, generation: int, parent_ids: list[str], operation: str):
        return self._eval(genotype, generation, parent_ids, operation,
                          parse_ok=self.last_parse_ok, mutator_tokens=self.last_mutator_tokens)

    def _remaining(self) -> int:
        return self.ledger.remaining

    @staticmethod
    def _gjson(g: dict) -> str:
        return to_json(g)
```

Registry:

```python
# src/optim/arms/__init__.py
from src.optim.arms.random_search import RandomSearch
from src.optim.arms.paraphrase import ParaphraseSearch
from src.optim.arms.bestofb import BestOfB
from src.optim.arms.ga import GA
from src.optim.arms.de import DE
from src.optim.arms.opro import OPRO
from src.optim.arms.protegi import ProTeGi
from src.optim.arms.gepa import GEPA
from src.optim.arms.promptbreeder import PromptBreeder
from src.optim.arms.mapelites import MAPElites

ARMS = {a.name: a for a in (RandomSearch, ParaphraseSearch, BestOfB, GA, DE, OPRO, ProTeGi, GEPA, PromptBreeder, MAPElites)}
```
(Add each import as its task lands; until then keep the registry to the classes that exist.)

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/optim/test_arms_base.py -q`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/optim/arms/base.py src/optim/arms/ops.py src/optim/arms/__init__.py tests/optim/test_arms_base.py
git commit -m "optim: arm base class, mutator and shared op templates"
```

---

### Task 2: Budget-matched baselines: random search, paraphrase-only, best-of-B

**Files:**
- Create: `src/optim/arms/random_search.py`, `src/optim/arms/paraphrase.py`, `src/optim/arms/bestofb.py`
- Test: `tests/optim/test_arms_baselines.py`

**Interfaces:**
- Consumes: `Arm`, `ops.MUTATE`, `ops.PARAPHRASE`, `ops.SYSTEM`.
- Produces: `RandomSearch.name == "random"`, `ParaphraseSearch.name == "paraphrase"` (returns the *last* paraphrase, not the best; records `best_on_train` in `writer.write_best` as usual so both are available), `BestOfB.name == "bestofb"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/optim/test_arms_baselines.py
import numpy as np, json, copy
from src.optim.arms.random_search import RandomSearch
from src.optim.arms.paraphrase import ParaphraseSearch
from src.optim.arms.bestofb import BestOfB
from src.optim.budget import BudgetLedger
from tests.optim.conftest import FakeFitness, FakeWriter, TEMPLATE

class CountingMutator:
    def __init__(self): self.n = 0
    def complete(self, system, user, temperature=0.7, max_tokens=1200):
        self.n += 1; g = copy.deepcopy(TEMPLATE); g["role_definition"] += " x" * (self.n % 7)
        return json.dumps(g), 10, 10

def _run(cls, B=12):
    fit, w, m = FakeFitness(), FakeWriter(), CountingMutator()
    arm = cls(cfg={"samples": B}, rng=np.random.default_rng(1), mutator=m, template=TEMPLATE)
    out = arm.run(fit, TEMPLATE, BudgetLedger(B=B), w)
    return out, w, m

def test_random_uses_exact_budget_and_returns_best():
    out, w, m = _run(RandomSearch)
    assert len(w.history) == 12 and m.n == 11          # seed counts as evaluation 0
    assert out == w.best

def test_paraphrase_returns_last_not_best():
    out, w, m = _run(ParaphraseSearch)
    assert len(w.history) == 12
    assert out == w.history[-1]["genotype"]

def test_bestofb_uses_exact_budget():
    out, w, m = _run(BestOfB)
    assert len(w.history) == 12 and out == w.best
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/optim/test_arms_baselines.py -q` → FAIL (modules missing)

- [ ] **Step 3: Implement the three arms**

```python
# src/optim/arms/random_search.py
from src.optim.arms.base import Arm
from src.optim.arms import ops

class RandomSearch(Arm):
    """B independent mutations of the seed; best on the optimisation panel. Separates 'search helps' from 'evolution helps'."""
    name = "random"
    def run(self, fitness, seed_genotype, ledger, writer):
        self._bind(fitness, ledger, writer)
        seed_id, _ = self._eval(seed_genotype, 0, [], "init")
        while self._remaining() > 0:
            g = self._ask(ops.SYSTEM, ops.MUTATE.format(genotype=self._gjson(seed_genotype)), fallback=seed_genotype)
            self._eval_asked(g, 1, [seed_id], "mutation")
        return self.best_genotype
```

```python
# src/optim/arms/paraphrase.py
from src.optim.arms.base import Arm
from src.optim.arms import ops

class ParaphraseSearch(Arm):
    """B chained paraphrases with NO selection pressure; returns the last one. The best-on-train is still logged."""
    name = "paraphrase"
    def run(self, fitness, seed_genotype, ledger, writer):
        self._bind(fitness, ledger, writer)
        cur, cur_id = seed_genotype, None
        cur_id, _ = self._eval(cur, 0, [], "init")
        while self._remaining() > 0:
            nxt = self._ask(ops.SYSTEM, ops.PARAPHRASE.format(genotype=self._gjson(cur)), fallback=cur)
            cur_id, _ = self._eval_asked(nxt, 1, [cur_id], "paraphrase")
            if self.last_parse_ok: cur = nxt
        return cur
```

```python
# src/optim/arms/bestofb.py
from src.optim.arms.base import Arm
from src.optim.arms import ops

class BestOfB(Arm):
    """Selection without evolution: B samples from the initial-population generator (a mutation at temperature 1.0), best on train."""
    name = "bestofb"
    def run(self, fitness, seed_genotype, ledger, writer):
        self._bind(fitness, ledger, writer)
        self.cfg = {**self.cfg, "temperature": 1.0}
        seed_id, _ = self._eval(seed_genotype, 0, [], "init")
        while self._remaining() > 0:
            g = self._ask(ops.SYSTEM, ops.MUTATE.format(genotype=self._gjson(seed_genotype)), fallback=seed_genotype)
            self._eval_asked(g, 0, [seed_id], "init_sample")
        return self.best_genotype
```

- [ ] **Step 4: Run tests** → 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/optim/arms/random_search.py src/optim/arms/paraphrase.py src/optim/arms/bestofb.py tests/optim/test_arms_baselines.py
git commit -m "optim: budget-matched baseline arms (random, paraphrase, best-of-B)"
```

---

### Task 3: GA (the NeurIPS method) and DE

**Files:**
- Create: `src/optim/arms/ga.py`, `src/optim/arms/de.py`
- Test: `tests/optim/test_arms_evolutionary.py`

**Interfaces:**
- Consumes: `Arm`, `ops.MUTATE`, `ops.CROSSOVER`, `ops.DE_STEP`. Config keys from `configs/optimizers/ga.yaml` (`population_size, generations, elites, tournament_k, crossover_rate, mutation_rate`) and `de.yaml` (`population_size, generations, crossover_rate`).
- Produces: `GA.name == "ga"`, `DE.name == "de"`. Both keep `self.population: list[tuple[str, float, dict]]` (candidate_id, score, genotype).

- [ ] **Step 1: Write the failing test**

```python
# tests/optim/test_arms_evolutionary.py
import numpy as np, json, copy
from src.optim.arms.ga import GA
from src.optim.arms.de import DE
from src.optim.budget import BudgetLedger
from tests.optim.conftest import FakeFitness, FakeWriter, TEMPLATE

class JitterMutator:
    def __init__(self, rng): self.rng = rng
    def complete(self, system, user, temperature=0.7, max_tokens=1200):
        g = copy.deepcopy(TEMPLATE); g["role_definition"] += " " + "y" * int(self.rng.integers(0, 40))
        return json.dumps(g), 10, 10

def test_ga_exhausts_budget_and_improves():
    rng = np.random.default_rng(3)
    cfg = {"population_size": 4, "generations": 4, "elites": 1, "tournament_k": 2, "crossover_rate": 0.8, "mutation_rate": 0.5}
    fit, w = FakeFitness(), FakeWriter()
    arm = GA(cfg, rng, JitterMutator(rng), TEMPLATE)
    best = arm.run(fit, TEMPLATE, BudgetLedger(B=16), w)
    assert len(w.history) == 16
    gens = [h["generation"] for h in w.history]; assert max(gens) == 3
    assert fit.evaluate(best).rps_cal_opt <= w.history[0]["rps_cal_opt"]

def test_de_exhausts_budget():
    rng = np.random.default_rng(4)
    fit, w = FakeFitness(), FakeWriter()
    arm = DE({"population_size": 4, "generations": 4, "crossover_rate": 0.8}, rng, JitterMutator(rng), TEMPLATE)
    arm.run(fit, TEMPLATE, BudgetLedger(B=16), w)
    assert len(w.history) == 16
    assert any(h["operation"] == "de_step" for h in w.history)
```

- [ ] **Step 2: Run to verify it fails** → FAIL (modules missing)

- [ ] **Step 3: Implement GA**

```python
# src/optim/arms/ga.py
import numpy as np
from src.optim.arms.base import Arm
from src.optim.arms import ops

class GA(Arm):
    """EvoPrompt-style GA on the JSON genotype: elitism, tournament selection, LLM crossover then LLM mutation.
    Budget: population_size evaluations per generation; generation 0 = seed + (pop-1) initial mutations."""
    name = "ga"
    def _init_population(self, seed_genotype):
        sid, s = self._eval(seed_genotype, 0, [], "init"); pop = [(sid, s, seed_genotype)]
        while len(pop) < int(self.cfg["population_size"]) and self._remaining() > 0:
            g = self._ask(ops.SYSTEM, ops.MUTATE.format(genotype=self._gjson(seed_genotype)), fallback=seed_genotype)
            cid, sc = self._eval_asked(g, 0, [sid], "init_mutation")
            if self.last_parse_ok: pop.append((cid, sc, g))
        return pop
    def _tournament(self, pop):
        k = int(self.cfg.get("tournament_k", 3)); idx = self.rng.choice(len(pop), size=min(k, len(pop)), replace=False)
        return min((pop[i] for i in idx), key=lambda t: t[1])
    def run(self, fitness, seed_genotype, ledger, writer):
        self._bind(fitness, ledger, writer)
        pop = self._init_population(seed_genotype)
        P, E = int(self.cfg["population_size"]), int(self.cfg.get("elites", 2))
        gen = 1
        while self._remaining() > 0:
            pop.sort(key=lambda t: t[1]); new = pop[:E]
            while len(new) < P and self._remaining() > 0:
                a, b = self._tournament(pop), self._tournament(pop)
                if self.rng.random() < float(self.cfg.get("crossover_rate", 0.8)):
                    child = self._ask(ops.SYSTEM, ops.CROSSOVER.format(a=self._gjson(a[2]), b=self._gjson(b[2])), fallback=a[2]); op = "crossover"
                else:
                    child, op = a[2], "copy"; self.last_parse_ok, self.last_mutator_tokens = True, (0, 0)
                if self.rng.random() < float(self.cfg.get("mutation_rate", 0.3)):
                    child = self._ask(ops.SYSTEM, ops.MUTATE.format(genotype=self._gjson(child)), fallback=child); op += "+mutation"
                cid, sc = self._eval_asked(child, gen, [a[0], b[0]], op)
                if self.last_parse_ok: new.append((cid, sc, child))
            pop = new if len(new) >= 2 else pop; gen += 1
        return self.best_genotype
```

- [ ] **Step 4: Implement DE**

```python
# src/optim/arms/de.py
from src.optim.arms.base import Arm
from src.optim.arms import ops
from src.optim.arms.ga import GA

class DE(GA):
    """EvoPrompt differential evolution: for each member x, pick distinct a,b,c; the LLM applies (a-b) to c and crosses with x; replace x if better."""
    name = "de"
    def run(self, fitness, seed_genotype, ledger, writer):
        self._bind(fitness, ledger, writer)
        pop = self._init_population(seed_genotype); gen = 1
        while self._remaining() > 0:
            for i in range(len(pop)):
                if self._remaining() == 0: break
                others = [j for j in range(len(pop)) if j != i]
                if len(others) < 3: break
                a, b, c = (pop[j] for j in self.rng.choice(others, size=3, replace=False))
                x = pop[i]
                trial = self._ask(ops.SYSTEM, ops.DE_STEP.format(a=self._gjson(a[2]), b=self._gjson(b[2]),
                                                                 c=self._gjson(c[2]), x=self._gjson(x[2])), fallback=x[2])
                cid, sc = self._eval_asked(trial, gen, [x[0], a[0], b[0], c[0]], "de_step")
                if self.last_parse_ok and sc < x[1]: pop[i] = (cid, sc, trial)
            gen += 1
        return self.best_genotype
```

- [ ] **Step 5: Run tests** → 2 passed. **Step 6: Commit**

```bash
git add src/optim/arms/ga.py src/optim/arms/de.py tests/optim/test_arms_evolutionary.py
git commit -m "optim: GA and DE arms on the JSON genotype"
```

---

### Task 4: OPRO and ProTeGi

**Files:**
- Create: `src/optim/arms/opro.py`, `src/optim/arms/protegi.py`
- Modify: `src/optim/fitness.py` — add `feedback(genotype, k: int) -> str` (plan 1 leaves a hook; see Interfaces)
- Test: `tests/optim/test_arms_reflective.py`

**Interfaces:**
- Consumes: `Fitness.feedback(genotype: dict, k: int = 10) -> str`: re-uses the last `evaluate()` call's beliefs (cached as `fitness.last_P`, `fitness.last_Y`) and returns `k` lines `"item <id> '<text>': model too high by <d> (mean predicted <m>, human <h>)"` sorted by absolute mean error. If `last_P` is missing it calls `evaluate` first (charging is the arm's responsibility: call `_eval` before `feedback`).
- Produces: `OPRO.name == "opro"`, `ProTeGi.name == "protegi"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/optim/test_arms_reflective.py
import numpy as np, json, copy
from src.optim.arms.opro import OPRO
from src.optim.arms.protegi import ProTeGi
from src.optim.budget import BudgetLedger
from tests.optim.conftest import FakeFitness, FakeWriter, TEMPLATE

class RecordingMutator:
    def __init__(self): self.users = []
    def complete(self, system, user, temperature=0.7, max_tokens=1200):
        self.users.append(user); g = copy.deepcopy(TEMPLATE); g["role_definition"] += " z" * (len(self.users) % 5)
        return json.dumps(g), 10, 10

def test_opro_shows_scored_history():
    fit, w, m = FakeFitness(), FakeWriter(), RecordingMutator()
    OPRO({"proposals_per_round": 2, "rounds": 3, "history_shown": 4}, np.random.default_rng(0), m, TEMPLATE).run(fit, TEMPLATE, BudgetLedger(B=7), w)
    assert len(w.history) == 7
    assert "score" in m.users[-1].lower() and "role_definition" in m.users[-1]

def test_protegi_uses_feedback_and_beam():
    fit, w, m = FakeFitness(), FakeWriter(), RecordingMutator()
    ProTeGi({"beam": 2, "rounds": 3, "gradients_per_candidate": 1, "edits_per_gradient": 1}, np.random.default_rng(0), m, TEMPLATE).run(fit, TEMPLATE, BudgetLedger(B=7), w)
    assert len(w.history) == 7
    assert any("FEEDBACK" in u for u in m.users)
```

`FakeFitness.feedback` in conftest returns `"item 1 'Worry about things.': model too high by 0.4"` repeated `k` times.

- [ ] **Step 2: Run to verify it fails** → FAIL

- [ ] **Step 3: Implement OPRO**

```python
# src/optim/arms/opro.py
from src.optim.arms.base import Arm
from src.optim.arms import ops

class OPRO(Arm):
    """LLM-as-optimizer: each round, show the best `history_shown` (persona, score) pairs, ask for one better persona, `proposals_per_round` times."""
    name = "opro"
    def _history_block(self, k):
        rows = sorted((h for h in self.history if h["parse_ok"]), key=lambda h: h["rps_cal_opt"])[:k]
        return "\n\n".join(f"score {h['rps_cal_opt']:.4f}\n{self._gjson(h['genotype'])}" for h in reversed(rows))
    def run(self, fitness, seed_genotype, ledger, writer):
        self._bind(fitness, ledger, writer)
        sid, _ = self._eval(seed_genotype, 0, [], "init"); rnd = 1
        while self._remaining() > 0:
            for _ in range(int(self.cfg.get("proposals_per_round", 8))):
                if self._remaining() == 0: break
                g = self._ask(ops.SYSTEM, ops.OPRO.format(history=self._history_block(int(self.cfg.get("history_shown", 12)))), fallback=self.best_genotype)
                self._eval_asked(g, rnd, [self.best_candidate_id], "opro_proposal")
            rnd += 1
        return self.best_genotype
```

- [ ] **Step 4: Implement ProTeGi**

```python
# src/optim/arms/protegi.py
from src.optim.arms.base import Arm
from src.optim.arms import ops

class ProTeGi(Arm):
    """Textual-gradient beam search: for each beam member, get feedback on its worst items, ask for an edited persona; keep the best `beam` candidates."""
    name = "protegi"
    def run(self, fitness, seed_genotype, ledger, writer):
        self._bind(fitness, ledger, writer)
        sid, s = self._eval(seed_genotype, 0, [], "init"); beam = [(sid, s, seed_genotype)]
        B, rnd = int(self.cfg.get("beam", 4)), 1
        while self._remaining() > 0:
            cands = list(beam)
            for cid, sc, g in beam:
                for _ in range(int(self.cfg.get("gradients_per_candidate", 2))):
                    if self._remaining() == 0: break
                    self.fitness.evaluate(g) if getattr(self.fitness, "last_genotype", None) is not g else None  # refresh cache without charging is NOT allowed: see note
                    fb = self.fitness.feedback(g, k=10)
                    for _ in range(int(self.cfg.get("edits_per_gradient", 1))):
                        if self._remaining() == 0: break
                        child = self._ask(ops.SYSTEM, ops.REFLECT.format(genotype=self._gjson(g), feedback=fb), fallback=g)
                        ncid, nsc = self._eval_asked(child, rnd, [cid], "textual_gradient_edit")
                        if self.last_parse_ok: cands.append((ncid, nsc, child))
            cands.sort(key=lambda t: t[1]); beam = cands[:B]; rnd += 1
        return self.best_genotype
```

Note on the marked line: `Fitness.feedback` must use the beliefs cached from the *charged* evaluation of `g`. Implement `Fitness.evaluate` to cache `(id(genotype_json), P, Y)` for the last 16 candidates keyed by `to_json(g)`, and `feedback` to look the key up; delete the marked line and rely on the cache (the beam members were all charged when evaluated). If the key is missing, `feedback` raises `KeyError` — that is a bug in the arm, not something to paper over with an uncharged evaluation.

- [ ] **Step 5: Run tests** → 2 passed. **Step 6: Commit**

```bash
git add src/optim/arms/opro.py src/optim/arms/protegi.py src/optim/fitness.py tests/optim/test_arms_reflective.py
git commit -m "optim: OPRO and ProTeGi arms; fitness feedback hook"
```

---

### Task 5: GEPA (reflective evolution with a Pareto pool)

**Files:**
- Create: `src/optim/arms/gepa.py`
- Test: `tests/optim/test_arms_gepa.py`

**Interfaces:**
- Consumes: `Fitness.evaluate_subset(genotype, respondent_idx: np.ndarray) -> FitnessResult` (plan 1; scores a minibatch of the optimisation panel; `forward_passes_per_candidate` scaled accordingly) and `Fitness.per_respondent_scores(genotype) -> np.ndarray` (per-respondent RPS_cal from the last full evaluation, used to build the Pareto pool). `BudgetLedger.charge_fraction(frac: float, ...)` (plan 1).
- Produces: `GEPA.name == "gepa"`. Algorithm (Agrawal et al. 2025): keep a pool of candidates with their per-respondent scores; the Pareto set = candidates that are best on at least one respondent; sample a parent from the Pareto set weighted by how many respondents it wins; run it on a minibatch, build feedback, reflect, propose a child; evaluate the child on the minibatch, and only if it improves there evaluate it on the full panel and add it to the pool. Optional: if `import gepa` succeeds and `cfg["library"] == "use"`, wrap `gepa.optimize` with an adapter whose `evaluate` calls `Fitness.evaluate_subset`; record `run_meta["gepa_impl"] = "library"`. Default is the built-in implementation (`"builtin"`).

- [ ] **Step 1: Write the failing test**

```python
# tests/optim/test_arms_gepa.py
import numpy as np, json, copy
from src.optim.arms.gepa import GEPA, pareto_front
from src.optim.budget import BudgetLedger
from tests.optim.conftest import FakeFitness, FakeWriter, TEMPLATE

def test_pareto_front_picks_per_respondent_winners():
    scores = {"a": np.array([0.1, 0.5, 0.5]), "b": np.array([0.5, 0.1, 0.5]), "c": np.array([0.6, 0.6, 0.6])}
    front, weights = pareto_front(scores)
    assert set(front) == {"a", "b"} and weights["a"] == weights["b"]

def test_gepa_budget_and_ops():
    class M:
        def complete(self, s, u, temperature=0.7, max_tokens=1200):
            g = copy.deepcopy(TEMPLATE); g["role_definition"] += " r"; return json.dumps(g), 10, 10
    fit, w = FakeFitness(), FakeWriter()
    GEPA({"minibatch_respondents": 4, "pareto_pool_max": 6, "reflection_items_shown": 5}, np.random.default_rng(0), M(), TEMPLATE).run(fit, TEMPLATE, BudgetLedger(B=10), w)
    assert w.ledger_used_at_end == 10
    assert any(h["operation"] == "gepa_reflect_full" for h in w.history)
```

`FakeWriter` must expose `ledger_used_at_end` set by the arm via `writer.finish(ledger)` (add `finish` to `CellWriter` in plan 1: writes `run_meta["candidate_evaluations"]`).

- [ ] **Step 2: Run to verify it fails** → FAIL

- [ ] **Step 3: Implement**

```python
# src/optim/arms/gepa.py
import numpy as np
from src.optim.arms.base import Arm
from src.optim.arms import ops

def pareto_front(scores: dict[str, np.ndarray]):
    """Candidates that are the best (lowest) on at least one respondent; weight = number of respondents won."""
    ids = list(scores); M = np.stack([scores[i] for i in ids])            # (C, n)
    winners = np.argmin(M, axis=0)
    counts = {ids[c]: int((winners == c).sum()) for c in range(len(ids))}
    front = [i for i in ids if counts[i] > 0]
    return front, {i: counts[i] for i in front}

class GEPA(Arm):
    name = "gepa"
    def run(self, fitness, seed_genotype, ledger, writer):
        self._bind(fitness, ledger, writer)
        n_opt = fitness.n_opt; mb = int(self.cfg.get("minibatch_respondents", 8)); frac = mb / n_opt
        pool: dict[str, tuple[dict, np.ndarray]] = {}
        sid, _ = self._eval(seed_genotype, 0, [], "init"); pool[sid] = (seed_genotype, fitness.per_respondent_scores(seed_genotype))
        rnd = 1
        while self._remaining() > 0:
            front, wts = pareto_front({k: v[1] for k, v in pool.items()})
            p = np.array([wts[k] for k in front], dtype=float); p /= p.sum()
            pid = front[self.rng.choice(len(front), p=p)]; parent = pool[pid][0]
            idx = self.rng.choice(n_opt, size=mb, replace=False)
            if self.ledger.remaining_fraction() < 2 * frac: break
            fitness.evaluate_subset(parent, idx); self.ledger.charge_fraction(frac, prompt_tokens=fitness.last_prompt_tokens)
            fb = fitness.feedback(parent, k=int(self.cfg.get("reflection_items_shown", 10)))
            child = self._ask(ops.SYSTEM, ops.REFLECT.format(genotype=self._gjson(parent), feedback=fb), fallback=parent)
            if not self.last_parse_ok:
                self._eval_asked(child, rnd, [pid], "gepa_reflect_unparsable"); rnd += 1; continue
            sub_child = fitness.evaluate_subset(child, idx); self.ledger.charge_fraction(frac, prompt_tokens=fitness.last_prompt_tokens)
            sub_parent = float(np.nanmean(pool[pid][1][idx]))
            if sub_child.rps_cal_opt < sub_parent and self._remaining() > 0:
                cid, _ = self._eval_asked(child, rnd, [pid], "gepa_reflect_full")
                pool[cid] = (child, fitness.per_respondent_scores(child))
                if len(pool) > int(self.cfg.get("pareto_pool_max", 12)):
                    keep = set(pareto_front({k: v[1] for k, v in pool.items()})[0]) | {self.best_candidate_id}
                    pool = {k: v for k, v in pool.items() if k in keep}
            rnd += 1
        # spend any leftover fraction on full evaluations of mutations of the best, so every arm exhausts B
        while self._remaining() > 0:
            g = self._ask(ops.SYSTEM, ops.MUTATE.format(genotype=self._gjson(self.best_genotype)), fallback=self.best_genotype)
            self._eval_asked(g, rnd, [self.best_candidate_id], "gepa_fill_mutation")
        writer.finish(ledger)
        return self.best_genotype
```

`BudgetLedger` (plan 1) must expose `remaining_fraction() -> float` (= `B - used_f`) and `charge_fraction(frac, prompt_tokens=0, generated_tokens=0)`; `used` is `ceil(used_f)`. All other arms call `charge` (frac = 1). Add `writer.finish(ledger)` calls at the end of every arm's `run` (Tasks 2–4 and 6–7) for consistency.

- [ ] **Step 4: Run tests** → 2 passed. **Step 5: Commit**

```bash
git add src/optim/arms/gepa.py tests/optim/test_arms_gepa.py
git commit -m "optim: GEPA arm (reflective evolution with a per-respondent Pareto pool)"
```

---

### Task 6: PromptBreeder

**Files:**
- Create: `src/optim/arms/promptbreeder.py`
- Test: `tests/optim/test_arms_promptbreeder.py`

**Interfaces:**
- Consumes: `Arm`, `ops.META_MUTATE`, `ops.SYSTEM`. Config: `population_size, generations, mutation_prompt_pool, hyper_mutation_rate`.
- Produces: `PromptBreeder.name == "promptbreeder"`; each population member is `(candidate_id, score, genotype, mutation_prompt: str)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/optim/test_arms_promptbreeder.py
import numpy as np, json, copy
from src.optim.arms.promptbreeder import PromptBreeder, SEED_MUTATION_PROMPTS
from src.optim.budget import BudgetLedger
from tests.optim.conftest import FakeFitness, FakeWriter, TEMPLATE

def test_promptbreeder_evolves_mutation_prompts():
    class M:
        def __init__(self): self.meta = 0
        def complete(self, s, u, temperature=0.7, max_tokens=1200):
            if "Write a different instruction" in u: self.meta += 1; return "Make each facet sentence more concrete.", 5, 5
            g = copy.deepcopy(TEMPLATE); g["role_definition"] += " p"; return json.dumps(g), 10, 10
    fit, w, m = FakeFitness(), FakeWriter(), M()
    PromptBreeder({"population_size": 4, "generations": 4, "mutation_prompt_pool": 4, "hyper_mutation_rate": 1.0}, np.random.default_rng(0), m, TEMPLATE).run(fit, TEMPLATE, BudgetLedger(B=16), w)
    assert len(w.history) == 16 and m.meta >= 1
    assert len(SEED_MUTATION_PROMPTS) >= 4
```

- [ ] **Step 2: Run to verify it fails** → FAIL

- [ ] **Step 3: Implement**

```python
# src/optim/arms/promptbreeder.py
from src.optim.arms.base import Arm
from src.optim.arms import ops
from src.optim.arms.ga import GA

SEED_MUTATION_PROMPTS = [
    "Rewrite the persona so each trait sentence names one observable habit.",
    "Sharpen the contrast between high and low levels of every facet.",
    "Make the role_definition a vivid first-person self-description.",
    "Replace abstract adjectives with concrete everyday behaviours.",
    "Tighten every sentence; remove hedging words.",
    "Add to critic_formulations one rule about answering as this specific person, not an average person.",
    "Reorder facets so the most distinctive ones come first.",
    "Rephrase each facet as what the person tends to do when stressed and when relaxed.",
]

class PromptBreeder(GA):
    """Fernando et al.: each member carries its own mutation prompt; with probability hyper_mutation_rate the mutation prompt itself is rewritten by the LLM before use."""
    name = "promptbreeder"
    def run(self, fitness, seed_genotype, ledger, writer):
        self._bind(fitness, ledger, writer)
        pool = list(SEED_MUTATION_PROMPTS[: int(self.cfg.get("mutation_prompt_pool", 8))])
        sid, s = self._eval(seed_genotype, 0, [], "init")
        pop = [(sid, s, seed_genotype, pool[0])]
        P = int(self.cfg["population_size"])
        while len(pop) < P and self._remaining() > 0:
            mp = pool[len(pop) % len(pool)]
            g = self._ask(ops.SYSTEM, f"{mp}\n\nPERSONA:\n{self._gjson(seed_genotype)}", fallback=seed_genotype)
            cid, sc = self._eval_asked(g, 0, [sid], "init_mutation")
            if self.last_parse_ok: pop.append((cid, sc, g, mp))
        gen = 1
        while self._remaining() > 0:
            pop.sort(key=lambda t: t[1]); new = pop[:1]
            while len(new) < P and self._remaining() > 0:
                a = self._tournament([t[:3] for t in pop]); parent = next(t for t in pop if t[0] == a[0])
                mp = parent[3]
                if self.rng.random() < float(self.cfg.get("hyper_mutation_rate", 0.25)):
                    text, pt, gt = self.mutator.complete(ops.SYSTEM, ops.META_MUTATE.format(mutation_prompt=mp), temperature=0.9, max_tokens=200)
                    if text.strip(): mp = text.strip()[:400]; pool.append(mp)
                child = self._ask(ops.SYSTEM, f"{mp}\n\nPERSONA:\n{self._gjson(parent[2])}", fallback=parent[2])
                cid, sc = self._eval_asked(child, gen, [parent[0]], "pb_mutation")
                if self.last_parse_ok: new.append((cid, sc, child, mp))
            pop = new if len(new) >= 2 else pop; gen += 1
        writer.finish(ledger)
        return self.best_genotype
```

- [ ] **Step 4: Run tests** → passed. **Step 5: Commit**

```bash
git add src/optim/arms/promptbreeder.py tests/optim/test_arms_promptbreeder.py
git commit -m "optim: PromptBreeder arm with self-referential mutation prompts"
```

---

### Task 7: MAP-Elites (GigaEvo-style quality-diversity), optional GigaEvo engine wrap

**Files:**
- Create: `src/optim/arms/mapelites.py`
- Test: `tests/optim/test_arms_mapelites.py`

**Interfaces:**
- Consumes: `Fitness.last_descriptor() -> dict` (plan 1: `{"prompt_length_chars": int, "predicted_sd": float}` from the last evaluated candidate; `predicted_sd` = mean over items of the belief's standard deviation). Config from `configs/optimizers/mapelites.yaml`.
- Produces: `MAPElites.name == "mapelites"`; `Archive` class with `add(cell_key, cid, score, genotype, descriptor)`, `sample(rng)`, `cells: dict`.
- GigaEvo: unzip `~/Downloads/gigaevo-core-internal-main.zip` next to the repo, read its README, and if it exposes a mutation operator that accepts (parent text, lineage insights) and returns text, implement `GigaEvoMutatorAdapter(Mutator)` that routes `ops.MUTATE` through it and set `run_meta["mapelites_engine"] = "gigaevo"`; otherwise `"builtin"`. The acceptance test below is the same either way.

- [ ] **Step 1: Write the failing test**

```python
# tests/optim/test_arms_mapelites.py
import numpy as np, json, copy
from src.optim.arms.mapelites import MAPElites, Archive, bin_index
from src.optim.budget import BudgetLedger
from tests.optim.conftest import FakeFitness, FakeWriter, TEMPLATE

def test_bin_index():
    assert bin_index(1200, [0, 1500, 2500, 4000, 100000]) == 0
    assert bin_index(3000, [0, 1500, 2500, 4000, 100000]) == 2
    assert bin_index(1e9, [0, 1500, 2500, 4000, 100000]) == 3

def test_archive_keeps_best_per_cell():
    a = Archive()
    a.add((0, 1), "c1", 0.20, {"x": 1}, {}); a.add((0, 1), "c2", 0.15, {"x": 2}, {}); a.add((1, 1), "c3", 0.30, {"x": 3}, {})
    assert a.cells[(0, 1)][0] == "c2" and len(a.cells) == 2

def test_mapelites_budget_and_islands():
    class M:
        def complete(self, s, u, temperature=0.7, max_tokens=1200):
            g = copy.deepcopy(TEMPLATE); g["role_definition"] += " m" * 3; return json.dumps(g), 10, 10
    fit, w = FakeFitness(), FakeWriter()
    cfg = {"islands": 2, "init_per_island": 2, "iterations": 8, "migrate_every": 4, "insight_window": 3,
           "archive_bins": {"prompt_length_chars": [0, 1500, 2500, 4000, 100000], "predicted_sd": [0, 0.8, 1.0, 1.2, 9]}}
    MAPElites(cfg, np.random.default_rng(0), M(), TEMPLATE).run(fit, TEMPLATE, BudgetLedger(B=12), w)
    assert len(w.history) == 12
    assert {h["island"] for h in w.history if "island" in h} == {0, 1}
```

`FakeFitness.last_descriptor()` returns `{"prompt_length_chars": len(json.dumps(g)), "predicted_sd": 0.9}`.

- [ ] **Step 2: Run to verify it fails** → FAIL

- [ ] **Step 3: Implement**

```python
# src/optim/arms/mapelites.py
import bisect
import numpy as np
from src.optim.arms.base import Arm
from src.optim.arms import ops

def bin_index(v, edges):
    return max(0, min(bisect.bisect_right(edges, v) - 1, len(edges) - 2))

class Archive:
    def __init__(self): self.cells: dict[tuple, tuple[str, float, dict, dict]] = {}
    def add(self, key, cid, score, genotype, descriptor):
        cur = self.cells.get(key)
        if cur is None or score < cur[1]: self.cells[key] = (cid, score, genotype, descriptor); return True
        return False
    def sample(self, rng):
        keys = list(self.cells); return self.cells[keys[rng.integers(len(keys))]]

class MAPElites(Arm):
    """Quality-diversity search in the spirit of GigaEvo/AlphaEvolve: per-island MAP-Elites archives over
    (prompt length, predicted dispersion), insight-conditioned mutation from lineage records, periodic migration."""
    name = "mapelites"
    def _key(self, d):
        b = self.cfg["archive_bins"]
        return tuple(bin_index(d[k], b[k]) for k in sorted(b))
    def _insights(self, lineage):
        w = int(self.cfg.get("insight_window", 6))
        return "\n".join(f"- {r['operation']}: score {r['rps_cal_opt']:.4f}, length {r.get('descriptor', {}).get('prompt_length_chars', '?')}, sd {r.get('descriptor', {}).get('predicted_sd', float('nan')):.2f}" for r in lineage[-w:])
    def _eval_d(self, g, gen, parents, op, island):
        cid, sc = self._eval_asked(g, gen, parents, op) if hasattr(self, "last_parse_ok") and op != "init" else self._eval(g, gen, parents, op)
        d = self.fitness.last_descriptor() if self.last_parse_ok else {"prompt_length_chars": len(self._gjson(g)), "predicted_sd": float("nan")}
        self.history[-1]["descriptor"] = d; self.history[-1]["island"] = island
        return cid, sc, d
    def run(self, fitness, seed_genotype, ledger, writer):
        self._bind(fitness, ledger, writer)
        n_isl = int(self.cfg.get("islands", 2)); archives = [Archive() for _ in range(n_isl)]; lineage = [[] for _ in range(n_isl)]
        self.last_parse_ok, self.last_mutator_tokens = True, (0, 0)
        for isl in range(n_isl):
            for i in range(int(self.cfg.get("init_per_island", 4))):
                if self._remaining() == 0: break
                if isl == 0 and i == 0:
                    self.last_parse_ok = True; cid, sc, d = self._eval_d(seed_genotype, 0, [], "init", isl); g = seed_genotype
                else:
                    g = self._ask(ops.SYSTEM, ops.MUTATE.format(genotype=self._gjson(seed_genotype)), fallback=seed_genotype)
                    cid, sc, d = self._eval_d(g, 0, [], "init_mutation", isl)
                if self.last_parse_ok: archives[isl].add(self._key(d), cid, sc, g, d)
                lineage[isl].append(self.history[-1])
        it = 1
        while self._remaining() > 0:
            isl = it % n_isl
            if not archives[isl].cells: archives[isl].add((0, 0), self.best_candidate_id, self.best_score, self.best_genotype, {})
            pid, psc, pg, pd = archives[isl].sample(self.rng)
            user = ops.MUTATE.format(genotype=self._gjson(pg)) + "\n\nLINEAGE INSIGHTS (recent attempts on this island, lower score is better):\n" + self._insights(lineage[isl])
            g = self._ask(ops.SYSTEM, user, fallback=pg)
            cid, sc, d = self._eval_d(g, it, [pid], "qd_mutation", isl)
            if self.last_parse_ok: archives[isl].add(self._key(d), cid, sc, g, d)
            lineage[isl].append(self.history[-1])
            if it % int(self.cfg.get("migrate_every", 12)) == 0 and n_isl > 1:
                src = archives[isl]; dst = archives[(isl + 1) % n_isl]
                best = min(src.cells.values(), key=lambda t: t[1]); dst.add(self._key(best[3]) if best[3] else (0, 0), *best)
            it += 1
        writer.finish(ledger)
        return self.best_genotype
```

- [ ] **Step 4: Run tests** → 3 passed. **Step 5: Commit**

```bash
git add src/optim/arms/mapelites.py tests/optim/test_arms_mapelites.py
git commit -m "optim: MAP-Elites quality-diversity arm (GigaEvo-style) with islands and lineage insights"
```

- [ ] **Step 6: GigaEvo engine check (no code unless it fits)**

```bash
cd ~/Development/personality-twins-arr && unzip -q -o ~/Downloads/gigaevo-core-internal-main.zip -d external_gigaevo
sed -n '1,120p' external_gigaevo/gigaevo-core-internal-main/README.md
grep -rn "def mutate\|class .*Mutation\|insight" external_gigaevo/gigaevo-core-internal-main --include=*.py | head -30
```
If a text-in/text-out mutation operator with lineage insights exists, add `GigaEvoMutatorAdapter` in `src/optim/arms/mapelites.py` implementing `Mutator.complete` by delegating `ops.MUTATE` requests to it, selectable with `cfg["gigaevo_engine"] == "use"`; the tests above must still pass with the adapter substituted. Record the choice in `run_meta.json` (`mapelites_engine`). If it does not fit, write one paragraph in `icml2027/STATUS.md` saying so and keep `"builtin"`.

---

### Task 8: Multi-objective variants (E5)

**Files:**
- Modify: `src/optim/arms/ga.py` (add `NSGA2` subclass), `src/optim/arms/mapelites.py` (objective vector), `src/optim/fitness.py` (expose `last_objectives()`)
- Create: `src/optim/arms/nsga2.py`
- Test: `tests/optim/test_arms_nsga2.py`

**Interfaces:**
- Consumes: `Fitness.last_objectives() -> dict` = `{"rps_cal": float, "neg_vr_between": float}` where `vr_between` is the between-respondent variance of the belief means divided by the human between-respondent variance on the optimisation panel (plan 1 computes both from the same beliefs; lower is better for both keys so dominance is uniform).
- Produces: `NSGA2.name == "nsga2"`, fast non-dominated sort and crowding distance (Deb et al. 2002); `run` returns the knee point of the final front (max distance from the line joining the front's extremes) and writes the whole front to `writer.write_front(list[dict])`.

- [ ] **Step 1: Write the failing test**

```python
# tests/optim/test_arms_nsga2.py
import numpy as np
from src.optim.arms.nsga2 import nondominated_sort, crowding_distance, knee_point

def test_nondominated_sort():
    F = np.array([[1, 5], [2, 3], [3, 1], [4, 4], [5, 5]], float)
    fronts = nondominated_sort(F)
    assert fronts[0] == [0, 1, 2] and fronts[1] == [3] and fronts[2] == [4]

def test_crowding_extremes_infinite():
    F = np.array([[1, 5], [2, 3], [3, 1]], float)
    d = crowding_distance(F, [0, 1, 2]); assert np.isinf(d[0]) and np.isinf(d[2]) and np.isfinite(d[1])

def test_knee_point():
    F = np.array([[0, 1], [0.2, 0.2], [1, 0]], float)
    assert knee_point(F, [0, 1, 2]) == 1
```

- [ ] **Step 2: Run to verify it fails** → FAIL

- [ ] **Step 3: Implement**

```python
# src/optim/arms/nsga2.py
import numpy as np
from src.optim.arms.ga import GA
from src.optim.arms import ops

def nondominated_sort(F):
    n = len(F); S = [[] for _ in range(n)]; cnt = np.zeros(n, int); fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q: continue
            if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]): S[p].append(q)
            elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]): cnt[p] += 1
        if cnt[p] == 0: fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                cnt[q] -= 1
                if cnt[q] == 0: nxt.append(q)
        i += 1; fronts.append(nxt)
    return [f for f in fronts if f]

def crowding_distance(F, front):
    d = {i: 0.0 for i in front}
    if len(front) <= 2: return {i: np.inf for i in front}
    for m in range(F.shape[1]):
        order = sorted(front, key=lambda i: F[i, m]); rng = F[order[-1], m] - F[order[0], m] or 1.0
        d[order[0]] = d[order[-1]] = np.inf
        for k in range(1, len(order) - 1):
            d[order[k]] += (F[order[k + 1], m] - F[order[k - 1], m]) / rng
    return d

def knee_point(F, front):
    P = F[front]; a, b = P[np.argmin(P[:, 0])], P[np.argmin(P[:, 1])]
    if np.allclose(a, b): return front[0]
    ab = b - a; ab /= np.linalg.norm(ab)
    dist = [np.linalg.norm((p - a) - np.dot(p - a, ab) * ab) for p in P]
    return front[int(np.argmax(dist))]

class NSGA2(GA):
    name = "nsga2"
    def _objectives(self):
        o = self.fitness.last_objectives(); return np.array([o["rps_cal"], o["neg_vr_between"]], float)
    def run(self, fitness, seed_genotype, ledger, writer):
        self._bind(fitness, ledger, writer)
        pop = []; objs = []
        base = self._init_population(seed_genotype)
        for cid, sc, g in base: pop.append((cid, sc, g)); objs.append(self._objectives_from_history(cid))
        P = int(self.cfg["population_size"]); gen = 1
        while self._remaining() > 0:
            children = []
            while len(children) < P and self._remaining() > 0:
                a, b = self._tournament(pop), self._tournament(pop)
                child = self._ask(ops.SYSTEM, ops.CROSSOVER.format(a=self._gjson(a[2]), b=self._gjson(b[2])), fallback=a[2])
                if self.rng.random() < float(self.cfg.get("mutation_rate", 0.3)):
                    child = self._ask(ops.SYSTEM, ops.MUTATE.format(genotype=self._gjson(child)), fallback=child)
                cid, sc = self._eval_asked(child, gen, [a[0], b[0]], "nsga2_offspring")
                if self.last_parse_ok: children.append((cid, sc, child)); objs.append(self._objectives()); self.history[-1]["objectives"] = objs[-1].tolist()
            allp = pop + children; F = np.array(objs[-len(allp):]) if len(objs) >= len(allp) else np.array([self._objectives_from_history(c[0]) for c in allp])
            fronts = nondominated_sort(F); keep = []
            for fr in fronts:
                if len(keep) + len(fr) <= P: keep += fr
                else:
                    cd = crowding_distance(F, fr); keep += sorted(fr, key=lambda i: -cd[i])[: P - len(keep)]; break
            pop = [allp[i] for i in keep]; objs = [F[i] for i in keep]; gen += 1
        F = np.array(objs); front = nondominated_sort(F)[0]
        writer.write_front([{"candidate_id": pop[i][0], "objectives": F[i].tolist(), "genotype": pop[i][2]} for i in front])
        writer.finish(ledger)
        return pop[knee_point(F, front)][2]
    def _objectives_from_history(self, cid):
        h = next(r for r in self.history if r["candidate_id"] == cid)
        if "objectives" not in h: h["objectives"] = list(self.fitness.last_objectives().values())
        return np.array(h["objectives"], float)
```

Register `NSGA2` in `ARMS`. `CellWriter.write_front` (plan 1) writes `pareto_front.json`.

- [ ] **Step 4: Run tests** → 3 passed. **Step 5: Commit**

```bash
git add src/optim/arms/nsga2.py src/optim/arms/__init__.py tests/optim/test_arms_nsga2.py
git commit -m "optim: NSGA-II multi-objective arm (calibrated RPS vs individuation)"
```

---

## Self-review

- Spec coverage: E1 arms GA, DE, GigaEvo/MAP-Elites, GEPA, OPRO, PromptBreeder, ProTeGi ✓; E3 LLM baselines random, paraphrase, best-of-B ✓ (hand-written base prompt and expert prompt are evaluated as candidates 0 in frozen_eval, plan 3); E5 multi-objective ✓ (NSGA-II; GEPA's pool; MAP-Elites descriptors). MIPROv2 is listed optional in the plan and omitted here deliberately.
- Placeholder scan: the only "optional" items are the GigaEvo engine wrap and the `gepa` library wrap, each with a concrete acceptance test and a recorded fallback.
- Type consistency: `Arm._eval -> (cid, score)`, `Arm._ask -> dict` with `last_parse_ok`/`last_mutator_tokens`, `Fitness.evaluate -> FitnessResult`, `Fitness.feedback -> str`, `Fitness.evaluate_subset`, `Fitness.per_respondent_scores`, `Fitness.last_descriptor`, `Fitness.last_objectives`, `BudgetLedger.charge/charge_fraction/remaining/remaining_fraction/used`, `CellWriter.append_history/write_best/write_front/finish` — all named identically in plan 1's Interfaces.
