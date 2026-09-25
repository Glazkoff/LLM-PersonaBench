# P0 Proper-Score Pipeline Implementation Plan (icml2027, plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `src/optim/` core that every optimizer arm and every experiment depends on: orientation-correct, proper-score (calibrated RPS) fitness on a belief readout, fixed respondent panels, exact budget accounting, and an atomic results writer.

**Architecture:** Pure-numpy scoring and calibration (`scoring.py`) is separated from I/O (`panels.py`, `results.py`) and from the model (`readout.py`). `Fitness` composes them: genotype → per-respondent system prompts (`persona.py`) → belief simplices (`readout.py`) → orientation flip (`orientation.py`) → calibrated RPS with α cross-fitted inside the optimisation panel. Arms (plan 2) see only `Fitness`.

**Tech Stack:** Python 3.11, numpy, pandas, openai client (vLLM OpenAI-compatible server with logprobs), pytest. Reuses `arr2026/scripts/hyp/_scoring.py` semantics and `src/utils/prompt.py` modifiers.

**Spec:** `icml2027/PLAN.md` §4 P0; `icml2027/PREREGISTRATION.md` §2–3; `icml2027/configs/panels.yaml`; `icml2027/results/README.md`.

## Global Constraints

- RPS is lower-is-better, normalised by K−1 = 4, identical to `arr2026/scripts/hyp/_scoring.rps`.
- Calibrated forecast: `F = Π_CDF[F₀ + α (Q − Q̄)]`, `Q` the model CDF at thresholds 1..4, `Q̄` its mean over the panel respondents, `F₀` the population CDF of the cluster's training block, `Π_CDF` = clip to [0,1] then isotonic (non-decreasing) projection per item. `α` grid from `panels.yaml`.
- During search, α is chosen by 2-fold cross-fitting inside the optimisation panel; the calibration and evaluation panels are never loaded by `Fitness`.
- Orientation: the model simplex is reversed (`P[..., ::-1]`) on items with `reverse == True` in `data/IPIP-NEO/120/item_key.csv` before any comparison with human answers.
- Items: per facet, the two lowest item ids are input (persona), the two highest are targets (scored). 60/60.
- Mass guard: `mass_on_scale` < 0.5 (mean over panel) marks a readout invalid.
- All writes atomic (`tmp` + `os.replace`). No `random` module; `numpy.random.default_rng(seed)` only.

---

### Task 1: Test fixtures and the orientation module

**Files:**
- Create: `src/optim/orientation.py`, `tests/optim/conftest.py`, `tests/optim/test_orientation.py`

**Interfaces:**
- Produces: `REVERSE_MASK_120: np.ndarray[bool] (120,)`; `flip_simplex(P: np.ndarray, item_ids: list[int]) -> np.ndarray` (reverses the last axis on reverse-keyed items, item ids are 1-based IPIP ids); `flip_answers(X: np.ndarray, item_ids) -> np.ndarray` (6 − x on reverse-keyed items). Conftest: `TEMPLATE: dict`, `FakeFitness`, `FakeWriter` (used by plan 2).

- [ ] **Step 1: Write the failing test**

```python
# tests/optim/test_orientation.py
import numpy as np
from src.optim.orientation import REVERSE_MASK_120, flip_simplex, flip_answers

def test_mask_has_55_reverse_items():
    assert REVERSE_MASK_120.shape == (120,) and int(REVERSE_MASK_120.sum()) == 55

def test_flip_simplex_only_on_reverse_items():
    ids = [1, 2]  # item 1 forward; find a reverse id for the check
    rev = int(np.where(REVERSE_MASK_120)[0][0]) + 1
    P = np.zeros((1, 2, 5)); P[0, 0, 0] = 1; P[0, 1, 0] = 1
    out = flip_simplex(P, [1, rev])
    assert out[0, 0, 0] == 1 and out[0, 1, 4] == 1

def test_flip_answers_is_involution():
    X = np.array([[1., 5., 3.]]); ids = [1, int(np.where(REVERSE_MASK_120)[0][0]) + 1, 3]
    assert np.array_equal(flip_answers(flip_answers(X, ids), ids), X)
```

- [ ] **Step 2: Run** `python -m pytest tests/optim/test_orientation.py -q` → FAIL (module missing)

- [ ] **Step 3: Implement**

```python
# src/optim/orientation.py
"""One orientation map for icml2027. The corpus stores human answers reverse-recoded on 55/120
items; the model answers the literal item. Every model-vs-human comparison goes through here."""
import csv
from pathlib import Path
import numpy as np

_KEY = Path(__file__).resolve().parents[2] / "data/IPIP-NEO/120/item_key.csv"

def _mask() -> np.ndarray:
    m = np.zeros(120, dtype=bool)
    with open(_KEY) as f:
        for r in csv.DictReader(f):
            if str(r["reverse"]).strip().lower() == "true":
                m[int(r["item"]) - 1] = True
    return m

REVERSE_MASK_120 = _mask()

def _sel(item_ids):
    return np.array([REVERSE_MASK_120[i - 1] for i in item_ids], dtype=bool)

def flip_simplex(P, item_ids):
    out = np.array(P, dtype=float, copy=True)
    s = _sel(item_ids)
    out[..., s, :] = out[..., s, ::-1]
    return out

def flip_answers(X, item_ids):
    out = np.array(X, dtype=float, copy=True)
    s = _sel(item_ids)
    out[..., s] = 6.0 - out[..., s]
    return out
```

```python
# tests/optim/conftest.py
import copy, json
from dataclasses import dataclass, field
import numpy as np
import pytest

TEMPLATE = {
    "role_definition": "You are a person answering a personality questionnaire.",
    "trait_formulations": {"openness": "You like new ideas.", "conscientiousness": "You are organised.",
                           "extraversion": "You enjoy company.", "agreeableness": "You are kind.",
                           "neuroticism": "You worry sometimes."},
    "facet_formulations": {"facet_anxiety": "You tend to worry.", "facet_orderliness": "You keep things tidy."},
    "critic_formulations": "Stay in character; answer as this person would.",
}

@dataclass
class _Res:
    rps_cal_opt: float; rps_raw_opt: float; s0_opt: float; mass: float
    prompt_tokens: int = 100; wall_s: float = 0.0; alphas: tuple = (0.5, 0.5)

class FakeFitness:
    """Deterministic stand-in: longer role_definition = better (lower) score."""
    forward_passes_per_candidate = 2400
    n_opt = 40
    last_prompt_tokens = 100
    def __init__(self): self._last = None
    def _score(self, g): return float(np.clip(0.20 - 0.001 * len(g["role_definition"]), 0.05, 0.5))
    def evaluate(self, g):
        self._last = g; s = self._score(g); return _Res(s, s + 0.02, -0.7, 0.99)
    def evaluate_subset(self, g, idx): return self.evaluate(g)
    def per_respondent_scores(self, g):
        s = self._score(g); return s + np.linspace(-0.01, 0.01, self.n_opt)
    def feedback(self, g, k=10): return "\n".join(["item 1 'Worry about things.': model too high by 0.4"] * k)
    def last_descriptor(self): return {"prompt_length_chars": len(json.dumps(self._last)), "predicted_sd": 0.9}
    def last_objectives(self): return {"rps_cal": self._score(self._last), "neg_vr_between": -0.2}

class FakeWriter:
    def __init__(self): self.history, self.best, self.front, self.ledger_used_at_end = [], None, None, None
    def append_history(self, rec): self.history.append(copy.deepcopy(rec))
    def write_best(self, g, rec): self.best = copy.deepcopy(g)
    def write_front(self, front): self.front = front
    def finish(self, ledger): self.ledger_used_at_end = ledger.used
```

- [ ] **Step 4: Run** → 3 passed. **Step 5: Commit**

```bash
git add src/optim/orientation.py tests/optim/conftest.py tests/optim/test_orientation.py
git commit -m "optim: single orientation map and shared test fixtures"
```

---

### Task 2: Scoring and calibration

**Files:**
- Create: `src/optim/scoring.py`, `tests/optim/test_scoring.py`

**Interfaces:**
- Produces: `rps(P, Y) -> (n, J)` per-cell RPS (NaN where Y NaN); `s0(P, Y)`; `cdf(P) -> (…, K-1)`; `pop_cdf(Y_train, K=5) -> (J, K-1)`; `iso_project(F) -> F` (monotone non-decreasing over thresholds, clipped [0,1]); `calibrate(Q, F0, alpha) -> F`; `cdf_to_simplex(F) -> P`; `select_alpha(Q, F0, Y, alphas) -> (alpha, score)`; `crossfit_rps_cal(Q, F0, Y, alphas, rng) -> (score, (a1, a2))`; `per_respondent(score_cells) -> (n,)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/optim/test_scoring.py
import numpy as np
from src.optim import scoring as S

def test_rps_matches_audit_definition():
    import sys; sys.path.insert(0, "arr2026/scripts/hyp")
    import _scoring as A
    rng = np.random.default_rng(0); P = rng.dirichlet(np.ones(5), size=(7, 3)); Y = rng.integers(1, 6, size=(7, 3)).astype(float)
    assert np.allclose(np.nanmean(S.rps(P, Y), axis=1), A.score("rps", P, Y))

def test_truth_beats_point_mass():
    p = np.array([.1, .2, .4, .2, .1]); rng = np.random.default_rng(1)
    Y = rng.choice(np.arange(1, 6), p=p, size=(20000, 1)).astype(float)
    truth = np.broadcast_to(p, (20000, 1, 5)); point = np.zeros_like(truth); point[..., 2] = 1
    assert np.nanmean(S.rps(truth, Y)) < np.nanmean(S.rps(point, Y))

def test_alpha_zero_returns_prior():
    rng = np.random.default_rng(2); Y = rng.integers(1, 6, size=(50, 4)).astype(float)
    F0 = S.pop_cdf(Y); Q = S.cdf(rng.dirichlet(np.ones(5), size=(50, 4)))
    F = S.calibrate(Q, F0, 0.0)
    assert np.allclose(F, np.broadcast_to(F0, F.shape))

def test_iso_projection_is_monotone():
    F = np.array([[[0.5, 0.3, 0.8, 1.2]]]); out = S.iso_project(F)
    assert np.all(np.diff(out, axis=-1) >= -1e-12) and out.min() >= 0 and out.max() <= 1

def test_crossfit_never_worse_than_prior_by_much_on_informative_Q():
    rng = np.random.default_rng(3); n, J = 80, 10
    Y = rng.integers(1, 6, size=(n, J)).astype(float)
    P = np.zeros((n, J, 5)); P[np.arange(n)[:, None], np.arange(J)[None, :], (Y - 1).astype(int)] = 0.6; P += 0.1; P /= P.sum(-1, keepdims=True)
    F0 = S.pop_cdf(Y); score, (a1, a2) = S.crossfit_rps_cal(S.cdf(P), F0, Y, [0, .5, 1, 1.5, 2], rng)
    prior = float(np.nanmean(S.rps(S.cdf_to_simplex(np.broadcast_to(F0, (n, J, 4))), Y)))
    assert score < prior and a1 > 0 and a2 > 0
```

- [ ] **Step 2: Run** → FAIL (module missing)

- [ ] **Step 3: Implement**

```python
# src/optim/scoring.py
import numpy as np

def cdf(P):
    return np.cumsum(P, axis=-1)[..., :-1]

def cdf_to_simplex(F):
    F = np.concatenate([np.zeros(F.shape[:-1] + (1,)), F, np.ones(F.shape[:-1] + (1,))], axis=-1)
    return np.clip(np.diff(F, axis=-1), 0, None)

def rps(P, Y):
    K = P.shape[-1]
    ind = np.stack([(Y <= t).astype(float) for t in range(1, K)], axis=-1)
    out = np.sum((cdf(P) - ind) ** 2, axis=-1) / (K - 1.0)
    return np.where(np.isnan(Y), np.nan, out)

def s0(P, Y):
    K = P.shape[-1]; lv = np.arange(1, K + 1, dtype=float)
    d = np.abs(lv - np.nan_to_num(Y, nan=1.0)[..., None]) / (K - 1.0)
    return np.where(np.isnan(Y), np.nan, -(1.0 - np.sum(P * d, axis=-1)))

def pop_cdf(Y_train, K=5):
    return np.stack([np.nanmean((Y_train <= t).astype(float) + np.where(np.isnan(Y_train), np.nan, 0), axis=0)
                     for t in range(1, K)], axis=-1)

def iso_project(F):
    """Pool-adjacent-violators per (respondent, item) over thresholds, then clip."""
    F = np.clip(np.asarray(F, float), 0, 1).copy(); flat = F.reshape(-1, F.shape[-1])
    for r in range(flat.shape[0]):
        v = list(flat[r]); w = [1.0] * len(v); i = 0
        vals, wts, lens = [], [], []
        for x in v:
            vals.append(x); wts.append(1.0); lens.append(1)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                tw = wts[-2] + wts[-1]; tv = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / tw; tl = lens[-2] + lens[-1]
                vals[-2:] = [tv]; wts[-2:] = [tw]; lens[-2:] = [tl]
        flat[r] = np.repeat(vals, lens)
    return F

def calibrate(Q, F0, alpha):
    Qbar = Q.mean(axis=0, keepdims=True)
    return iso_project(F0[None] + alpha * (Q - Qbar))

def _cal_score(Q, F0, Y, alpha):
    return float(np.nanmean(rps(cdf_to_simplex(calibrate(Q, F0, alpha)), Y)))

def select_alpha(Q, F0, Y, alphas):
    scores = [(_cal_score(Q, F0, Y, a), a) for a in alphas]
    s, a = min(scores); return float(a), s

def crossfit_rps_cal(Q, F0, Y, alphas, rng):
    n = Q.shape[0]; perm = rng.permutation(n); A, B = perm[: n // 2], perm[n // 2:]
    aA, _ = select_alpha(Q[A], F0, Y[A], alphas); aB, _ = select_alpha(Q[B], F0, Y[B], alphas)
    # alpha selected on one half scores the other; Qbar uses the half being scored
    sB = _cal_score(Q[B], F0, Y[B], aA); sA = _cal_score(Q[A], F0, Y[A], aB)
    return (sA * len(A) + sB * len(B)) / n, (aA, aB)

def per_respondent(cells):
    return np.nanmean(cells, axis=-1)
```

Note: `iso_project` is O(n·J·K) in Python; fine for n ≤ 200, J = 60. If profiling in E0 shows it dominates, vectorise with the closed form for K−1 = 4 thresholds.

- [ ] **Step 4: Run** → 5 passed. **Step 5: Commit**

```bash
git add src/optim/scoring.py tests/optim/test_scoring.py
git commit -m "optim: RPS, S0, population CDF, isotonic projection, calibrated RPS with cross-fitted alpha"
```

---

### Task 3: Genotype parsing and budget ledger

**Files:**
- Create: `src/optim/genotype.py`, `src/optim/budget.py`, `tests/optim/test_genotype_budget.py`

**Interfaces:**
- Produces: `GenotypeParseError(ValueError)`; `parse_genotype(text: str, template: dict) -> dict` (extracts the first `{...}` block, requires the four keys, keeps only keys present in the template's `trait_formulations`/`facet_formulations`, fills missing entries from the template, rejects if > 50% of entries missing); `to_json(g) -> str` (sorted keys, stable); `seed_genotype(cluster: int) -> dict` (from `src/prompt/previous_base_prompt/{traits,facets}.json` + `src/prompt/system.json`). `BudgetExhausted(RuntimeError)`; `BudgetLedger(B: int)` with `.charge(passes, prompt_tokens=0, generated_tokens=0)`, `.charge_fraction(frac, …)`, `.used: int` (= ceil of fractional use), `.remaining: int`, `.remaining_fraction() -> float`, `.forward_passes`, `.prompt_tokens`, `.generated_tokens`, `.to_dict()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/optim/test_genotype_budget.py
import json, pytest
from src.optim.genotype import parse_genotype, GenotypeParseError, to_json, seed_genotype
from src.optim.budget import BudgetLedger, BudgetExhausted
from tests.optim.conftest import TEMPLATE

def test_parse_wrapped_json_and_fill_missing():
    g = dict(TEMPLATE); g = {**g, "trait_formulations": {**g["trait_formulations"]}}; del g["trait_formulations"]["openness"]
    out = parse_genotype("Sure! ```json\n" + json.dumps(g) + "\n```", TEMPLATE)
    assert out["trait_formulations"]["openness"] == TEMPLATE["trait_formulations"]["openness"]

def test_parse_rejects_garbage():
    with pytest.raises(GenotypeParseError): parse_genotype("no json", TEMPLATE)

def test_parse_drops_unknown_keys():
    g = json.loads(json.dumps(TEMPLATE)); g["facet_formulations"]["facet_made_up"] = "x"
    assert "facet_made_up" not in parse_genotype(json.dumps(g), TEMPLATE)["facet_formulations"]

def test_seed_genotype_has_all_keys():
    g = seed_genotype(0)
    assert set(g) >= {"role_definition", "trait_formulations", "facet_formulations", "critic_formulations", "intensity_modifiers"}

def test_budget():
    L = BudgetLedger(B=3); L.charge(10); L.charge_fraction(0.2); L.charge_fraction(0.2)
    assert L.used == 2 and L.remaining == 1 and abs(L.remaining_fraction() - 1.6) < 1e-9
    L.charge(1)
    with pytest.raises(BudgetExhausted): L.charge(1)
```

- [ ] **Step 2: Run** → FAIL

- [ ] **Step 3: Implement**

```python
# src/optim/genotype.py
import copy, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEYS = ("role_definition", "trait_formulations", "facet_formulations", "critic_formulations")

class GenotypeParseError(ValueError): pass

def to_json(g: dict) -> str:
    return json.dumps({k: g[k] for k in KEYS if k in g}, sort_keys=True, ensure_ascii=False, indent=1)

def _first_object(text: str) -> dict:
    s = text.find("{")
    while s != -1:
        depth = 0
        for i in range(s, len(text)):
            depth += text[i] == "{"; depth -= text[i] == "}"
            if depth == 0:
                try: return json.loads(text[s:i + 1])
                except json.JSONDecodeError: break
        s = text.find("{", s + 1)
    raise GenotypeParseError("no JSON object found")

def parse_genotype(text: str, template: dict) -> dict:
    obj = _first_object(text or "")
    if not isinstance(obj, dict) or any(k not in obj for k in KEYS):
        raise GenotypeParseError("missing required keys")
    out = copy.deepcopy(template); missing = total = 0
    for k in ("trait_formulations", "facet_formulations"):
        src = obj.get(k) if isinstance(obj.get(k), dict) else {}
        for name in template[k]:
            total += 1
            v = src.get(name)
            if isinstance(v, str) and v.strip(): out[k][name] = v.strip()
            else: missing += 1
    if total and missing / total > 0.5: raise GenotypeParseError("more than half the entries missing")
    for k in ("role_definition", "critic_formulations"):
        if not isinstance(obj[k], str) or not obj[k].strip(): raise GenotypeParseError(f"{k} empty")
        out[k] = obj[k].strip()
    return out

def seed_genotype(cluster: int) -> dict:
    sysm = json.loads((ROOT / "src/prompt/system.json").read_text())
    tr = json.loads((ROOT / "src/prompt/previous_base_prompt/traits.json").read_text())[str(cluster)]
    fc = json.loads((ROOT / "src/prompt/previous_base_prompt/facets.json").read_text())[str(cluster)]
    return {"role_definition": sysm["role"], "trait_formulations": dict(tr), "facet_formulations": dict(fc),
            "critic_formulations": sysm["critic_internal"], "intensity_modifiers": sysm["intensity_modifiers"]}
```

```python
# src/optim/budget.py
import math

class BudgetExhausted(RuntimeError): pass

class BudgetLedger:
    def __init__(self, B: int):
        self.B = int(B); self.used_f = 0.0; self.forward_passes = 0; self.prompt_tokens = 0; self.generated_tokens = 0
    @property
    def used(self) -> int: return math.ceil(self.used_f - 1e-9)
    @property
    def remaining(self) -> int: return max(0, self.B - self.used)
    def remaining_fraction(self) -> float: return max(0.0, self.B - self.used_f)
    def charge_fraction(self, frac, passes=0, prompt_tokens=0, generated_tokens=0):
        if self.used_f + frac > self.B + 1e-9: raise BudgetExhausted(f"B={self.B} exhausted")
        self.used_f += frac; self.forward_passes += int(passes); self.prompt_tokens += int(prompt_tokens); self.generated_tokens += int(generated_tokens)
    def charge(self, passes, prompt_tokens=0, generated_tokens=0):
        self.charge_fraction(1.0, passes, prompt_tokens, generated_tokens)
    def to_dict(self):
        return {"budget_B": self.B, "candidate_evaluations": self.used, "candidate_evaluations_exact": round(self.used_f, 4),
                "forward_passes": self.forward_passes, "prompt_tokens": self.prompt_tokens, "generated_tokens": self.generated_tokens}
```

The GEPA arm (plan 2) must round its fractional minibatch charges so the final `used_f == B` exactly; its fill loop does this because `charge` raises before exceeding B.

- [ ] **Step 4: Run** → 5 passed. **Step 5: Commit**

```bash
git add src/optim/genotype.py src/optim/budget.py tests/optim/test_genotype_budget.py
git commit -m "optim: genotype parser with repair and a strict budget ledger"
```

---

### Task 4: Panels and item split

**Files:**
- Create: `src/optim/panels.py`, `tests/optim/test_panels.py`

**Interfaces:**
- Produces: `ITEM_SPLIT: tuple[list[int], list[int]]` (input_ids, target_ids; 60 each); `@dataclass Panels(cluster, seed, train: pd.DataFrame, opt, cal, eval)`; `load_corpus(path) -> pd.DataFrame`; `excluded_cases(globs: list[str]) -> set[int]`; `make_panels(df, cluster, seed, sizes: dict, exclude: set[int]) -> Panels` (disjoint, deterministic, drawn with `default_rng(seed * 1000 + cluster)`); `persona_scores(row_answers_input: np.ndarray) -> dict[str, float]` (30 facet scores on 0–100 from the input half, human orientation; see Step 3); `answers(df, ids) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

```python
# tests/optim/test_panels.py
import numpy as np, pandas as pd
from src.optim.panels import ITEM_SPLIT, make_panels, persona_scores, FACET_OF_ITEM

def _toy(n=2000):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.integers(1, 6, size=(n, 120)), columns=[f"i{i}" for i in range(1, 121)])
    df["case"] = np.arange(n); df["clusters"] = rng.integers(0, 4, size=n); return df

def test_item_split_is_60_60_and_two_per_facet():
    inp, tgt = ITEM_SPLIT
    assert len(inp) == 60 and len(tgt) == 60 and not set(inp) & set(tgt)
    from collections import Counter
    assert set(Counter(FACET_OF_ITEM[i] for i in inp).values()) == {2}

def test_panels_disjoint_deterministic():
    df = _toy(); sizes = {"train": 200, "opt": 40, "cal": 40, "eval": 100}
    a = make_panels(df, 1, 7, sizes, exclude={0, 1, 2}); b = make_panels(df, 1, 7, sizes, exclude={0, 1, 2})
    ids = [set(p["case"]) for p in (a.train, a.opt, a.cal, a.eval)]
    assert sum(len(s) for s in ids) == len(set().union(*ids))
    assert list(a.eval["case"]) == list(b.eval["case"]) and not ({0, 1, 2} & set().union(*ids))
    assert (a.eval["clusters"] == 1).all()

def test_persona_scores_range():
    s = persona_scores(np.full(60, 5.0))
    assert len(s) == 30 and all(v == 100.0 for v in s.values())
```

- [ ] **Step 2: Run** → FAIL

- [ ] **Step 3: Implement**

```python
# src/optim/panels.py
import csv, glob, json
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]

def _key():
    with open(ROOT / "data/IPIP-NEO/120/item_key.csv") as f:
        return list(csv.DictReader(f))

_KEYROWS = _key()
FACET_OF_ITEM = {int(r["item"]): r["facet_key"] for r in _KEYROWS}
FACET_NAME = {r["facet_key"]: r["facet"] for r in _KEYROWS}
TEXT_OF_ITEM = {int(r["item"]): r["text"] for r in _KEYROWS}

def _split():
    by = {}
    for i in sorted(FACET_OF_ITEM): by.setdefault(FACET_OF_ITEM[i], []).append(i)
    inp = sorted(i for items in by.values() for i in items[:2]); tgt = sorted(i for items in by.values() for i in items[2:])
    return inp, tgt

ITEM_SPLIT = _split()

@dataclass
class Panels:
    cluster: int; seed: int
    train: pd.DataFrame; opt: pd.DataFrame; cal: pd.DataFrame; eval: pd.DataFrame

def load_corpus(path=ROOT / "data/raw/df_ipipneo_120_clusters") -> pd.DataFrame:
    return pd.read_csv(path)

def excluded_cases(globs) -> set:
    out = set()
    for pat in globs:
        for p in glob.glob(str(ROOT / pat), recursive=True):
            d = json.load(open(p)); [out.update(int(c) for c in v) for v in d.get("splits", {}).values()]
    return out

def make_panels(df, cluster, seed, sizes, exclude=frozenset()) -> Panels:
    pool = df[(df["clusters"] == cluster) & (~df["case"].isin(exclude))]
    pool = pool.dropna(subset=[f"i{i}" for i in range(1, 121)])
    rng = np.random.default_rng(seed * 1000 + cluster)
    need = sizes["train"] + sizes["opt"] + sizes["cal"] + sizes["eval"]
    if len(pool) < need: raise ValueError(f"cluster {cluster}: {len(pool)} < {need}")
    idx = rng.permutation(len(pool))[:need]; rows = pool.iloc[idx]
    a = sizes["eval"]; b = a + sizes["cal"]; c = b + sizes["opt"]
    # eval drawn first so it never depends on train size
    return Panels(cluster, seed, train=rows.iloc[c:], opt=rows.iloc[b:c], cal=rows.iloc[a:b], eval=rows.iloc[:a])

def answers(df, ids) -> np.ndarray:
    return df[[f"i{i}" for i in ids]].to_numpy(float)

def persona_scores(x_input) -> dict:
    """30 facet scores on 0-100 from the input half (human orientation, i.e. the corpus's recoded scale):
    mean of the facet's two input answers mapped 1..5 -> 0..100."""
    inp = ITEM_SPLIT[0]; out = {}
    for fk in sorted(set(FACET_OF_ITEM.values())):
        vals = [x_input[k] for k, i in enumerate(inp) if FACET_OF_ITEM[i] == fk]
        out[fk] = float((np.mean(vals) - 1.0) / 4.0 * 100.0)
    return out
```

- [ ] **Step 4: Run** → 3 passed. **Step 5: Commit**

```bash
git add src/optim/panels.py tests/optim/test_panels.py
git commit -m "optim: deterministic disjoint panels, 60/60 item split, input-half persona scores"
```

---

### Task 5: Persona prompt builder and belief readout

**Files:**
- Create: `src/optim/persona.py`, `src/optim/readout.py`, `tests/optim/test_persona_readout.py`

**Interfaces:**
- Consumes: `panels.persona_scores`, `panels.FACET_NAME`, `panels.TEXT_OF_ITEM`, `src/utils/prompt.get_modifier_bisect`.
- Produces: `system_prompt(genotype: dict, scores: dict[str, float]) -> str` (trait lines from the five trait facets' means; facet lines for every facet in `genotype["facet_formulations"]` matched by facet name, using `get_modifier_bisect` with `genotype["intensity_modifiers"]`); `ITEM_QUESTION = "How accurately does this statement describe you? \"{text}\"\nAnswer with a single number from 1 (very inaccurate) to 5 (very accurate)."`; `class Readout(Protocol): beliefs(systems: list[str], item_ids: list[int]) -> tuple[np.ndarray (n,J,5), np.ndarray mass (n,J), int prompt_tokens]`; `class VLLMReadout(Readout)(model, base_url, prefill, workers=48)`; `class FakeReadout(Readout)(rng)` for tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/optim/test_persona_readout.py
import numpy as np
from src.optim.persona import system_prompt
from src.optim.readout import FakeReadout
from src.optim.genotype import seed_genotype
from src.optim.panels import persona_scores

def test_system_prompt_contains_modifiers_and_role():
    g = seed_genotype(0); s = system_prompt(g, persona_scores(np.full(60, 1.0)))
    assert g["role_definition"][:20] in s and "very little" in s

def test_system_prompt_differs_by_person():
    g = seed_genotype(0)
    assert system_prompt(g, persona_scores(np.full(60, 1.0))) != system_prompt(g, persona_scores(np.full(60, 5.0)))

def test_fake_readout_shapes():
    P, mass, tok = FakeReadout(np.random.default_rng(0)).beliefs(["a", "b"], [1, 2, 3])
    assert P.shape == (2, 3, 5) and np.allclose(P.sum(-1), 1) and mass.shape == (2, 3)
```

- [ ] **Step 2: Run** → FAIL

- [ ] **Step 3: Implement**

```python
# src/optim/persona.py
import numpy as np
from src.utils.prompt import get_modifier_bisect
from src.optim.panels import FACET_NAME

TRAIT_OF = {"N": "neuroticism", "E": "extraversion", "O": "openness", "A": "agreeableness", "C": "conscientiousness"}

def _norm(s): return s.lower().replace("facet_", "").replace("-", "_").replace(" ", "_")

def system_prompt(genotype: dict, scores: dict) -> str:
    mods = genotype["intensity_modifiers"]
    trait_val = {TRAIT_OF[t]: float(np.mean([v for fk, v in scores.items() if fk[0] == t])) for t in TRAIT_OF}
    by_name = {_norm(FACET_NAME[fk]): v for fk, v in scores.items()}
    lines = [f"- This trait ({t}) describes you {get_modifier_bisect(trait_val[t], mods)}: {d}"
             for t, d in genotype["trait_formulations"].items() if t in trait_val]
    for f, d in genotype["facet_formulations"].items():
        v = by_name.get(_norm(f))
        if v is not None: lines.append(f"- This facet ({f}) describes you {get_modifier_bisect(v, mods)}: {d}")
    return (f"{genotype['role_definition']}\nYour profile:\n" + "\n".join(lines) +
            f"\n\nInternal reflection guideline:\n{genotype['critic_formulations']}")
```

```python
# src/optim/readout.py
import math, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from src.optim.panels import TEXT_OF_ITEM

DIGITS = ["1", "2", "3", "4", "5"]
ITEM_QUESTION = ('How accurately does this statement describe you? "{text}"\n'
                 "Answer with a single number from 1 (very inaccurate) to 5 (very accurate).")

class FakeReadout:
    def __init__(self, rng): self.rng = rng
    def beliefs(self, systems, item_ids):
        n, J = len(systems), len(item_ids)
        return self.rng.dirichlet(np.ones(5), size=(n, J)), np.full((n, J), 0.99), 100 * n * J

class VLLMReadout:
    """Belief readout from the first answer token's logprobs; thinking off, assistant turn prefilled.
    Prefix caching makes the shared system prompt cheap across a respondent's items."""
    def __init__(self, model, base_url=None, prefill="My answer is ", workers=48):
        from openai import OpenAI
        self.model, self.prefill, self.workers = model, prefill, workers
        self.client = OpenAI(base_url=base_url or os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
                             api_key=os.environ.get("LOCAL_LLM_API_KEY", "EMPTY"), timeout=600)
    def _one(self, system, item):
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=1, temperature=0.0, logprobs=True, top_logprobs=20,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": ITEM_QUESTION.format(text=TEXT_OF_ITEM[item])},
                      {"role": "assistant", "content": self.prefill}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False},
                        "add_generation_prompt": False, "continue_final_message": True})
        top = r.choices[0].logprobs.content[0].top_logprobs
        p = np.zeros(5)
        for t in top:
            tok = t.token.strip()
            if tok in DIGITS: p[int(tok) - 1] += math.exp(t.logprob)
        mass = float(p.sum()); p = p / mass if mass > 0 else np.full(5, 0.2)
        return p, mass, int(getattr(r.usage, "prompt_tokens", 0) or 0)
    def beliefs(self, systems, item_ids):
        n, J = len(systems), len(item_ids); P = np.zeros((n, J, 5)); M = np.zeros((n, J)); tok = 0
        jobs = [(a, b) for a in range(n) for b in range(J)]
        with ThreadPoolExecutor(self.workers) as ex:
            for (a, b), (p, m, t) in zip(jobs, ex.map(lambda ab: self._one(systems[ab[0]], item_ids[ab[1]]), jobs)):
                P[a, b], M[a, b] = p, m; tok += t
        return P, M, tok
```

- [ ] **Step 4: Run** → 3 passed. **Step 5: Commit**

```bash
git add src/optim/persona.py src/optim/readout.py tests/optim/test_persona_readout.py
git commit -m "optim: individual persona builder and vLLM logprob belief readout"
```

---

### Task 6: Fitness

**Files:**
- Create: `src/optim/fitness.py`, `tests/optim/test_fitness.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: `@dataclass FitnessResult(rps_cal_opt, rps_raw_opt, s0_opt, mass, prompt_tokens, wall_s, alphas: tuple)`; `class Fitness(panels: Panels, readout, alphas: list[float], rng, objective: str = "rps_cal")` with `n_opt`, `forward_passes_per_candidate (= n_opt * 60)`, `last_prompt_tokens`, `evaluate(g) -> FitnessResult` (when `objective == "s0"`, `rps_cal_opt` holds the S₀ loss so arms stay objective-agnostic; the true RPS_cal is still logged), `evaluate_subset(g, idx) -> FitnessResult`, `per_respondent_scores(g) -> (n_opt,)`, `feedback(g, k) -> str`, `last_descriptor() -> dict`, `last_objectives() -> dict`. Fitness never touches `panels.cal` or `panels.eval` (enforced by the test).

- [ ] **Step 1: Write the failing test**

```python
# tests/optim/test_fitness.py
import numpy as np, pandas as pd, pytest
from src.optim.fitness import Fitness
from src.optim.panels import make_panels
from src.optim.readout import FakeReadout
from src.optim.genotype import seed_genotype

def _panels():
    rng = np.random.default_rng(0); n = 800
    df = pd.DataFrame(rng.integers(1, 6, size=(n, 120)).astype(float), columns=[f"i{i}" for i in range(1, 121)])
    df["case"] = np.arange(n); df["clusters"] = 0
    return make_panels(df, 0, 1, {"train": 400, "opt": 40, "cal": 40, "eval": 100})

class Tripwire(pd.DataFrame):
    @property
    def _constructor(self): return Tripwire
    def __getitem__(self, k): raise AssertionError("eval/cal panel touched during search")

def test_evaluate_and_isolation():
    p = _panels(); p.cal = Tripwire(p.cal); p.eval = Tripwire(p.eval)
    f = Fitness(p, FakeReadout(np.random.default_rng(1)), [0, .5, 1], np.random.default_rng(2))
    r = f.evaluate(seed_genotype(0))
    assert 0 < r.rps_cal_opt < 1 and f.forward_passes_per_candidate == 40 * 60
    assert f.per_respondent_scores(seed_genotype(0)).shape == (40,)
    assert "model too" in f.feedback(seed_genotype(0), k=3)
    assert set(f.last_objectives()) == {"rps_cal", "neg_vr_between"}
    sub = f.evaluate_subset(seed_genotype(0), np.arange(8)); assert 0 < sub.rps_cal_opt < 1
```

- [ ] **Step 2: Run** → FAIL

- [ ] **Step 3: Implement**

```python
# src/optim/fitness.py
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
    rps_cal_opt: float; rps_raw_opt: float; s0_opt: float; mass: float
    prompt_tokens: int; wall_s: float; alphas: tuple

class Fitness:
    def __init__(self, panels, readout, alphas, rng, objective="rps_cal"):
        self.readout, self.alphas, self.rng, self.objective = readout, list(alphas), rng, objective
        inp, self.tgt = ITEM_SPLIT
        self.F0 = S.pop_cdf(answers(panels.train, self.tgt))
        self.Y = answers(panels.opt, self.tgt)
        self.scores = [persona_scores(x) for x in answers(panels.opt, inp)]
        self.human_between_var = np.nanvar(self.Y, axis=0).mean()
        self.n_opt = len(self.Y); self.forward_passes_per_candidate = self.n_opt * len(self.tgt)
        self.last_prompt_tokens = 0; self._cache = {}; self._last_key = None

    def _beliefs(self, g, idx):
        systems = [system_prompt(g, self.scores[i]) for i in idx]
        P, M, tok = self.readout.beliefs(systems, self.tgt)
        self.last_prompt_tokens = tok
        return flip_simplex(P, self.tgt), M

    def _result(self, P, M, Y, t0):
        Q = S.cdf(P)
        cal, alphas = S.crossfit_rps_cal(Q, self.F0, Y, self.alphas, self.rng)
        raw = float(np.nanmean(S.rps(P, Y))); s0v = float(np.nanmean(S.s0(P, Y)))
        primary = s0v if self.objective == "s0" else cal
        return FitnessResult(float(primary), raw, s0v, float(M.mean()), self.last_prompt_tokens,
                             time.perf_counter() - t0, tuple(alphas)), cal

    def evaluate(self, g) -> FitnessResult:
        t0 = time.perf_counter(); P, M = self._beliefs(g, range(self.n_opt))
        res, cal = self._result(P, M, self.Y, t0)
        key = to_json(g); self._last_key = key
        a = float(np.mean(res.alphas))
        per = S.per_respondent(S.rps(S.cdf_to_simplex(S.calibrate(S.cdf(P), self.F0, a)), self.Y))
        self._cache[key] = {"P": P, "per": per, "cal": cal}
        if len(self._cache) > 16: self._cache.pop(next(iter(self._cache)))
        return res

    def evaluate_subset(self, g, idx) -> FitnessResult:
        t0 = time.perf_counter(); idx = np.asarray(idx); P, M = self._beliefs(g, idx)
        return self._result(P, M, self.Y[idx], t0)[0]

    def per_respondent_scores(self, g):
        return self._cache[to_json(g)]["per"]

    def feedback(self, g, k=10) -> str:
        P = self._cache[to_json(g)]["P"]; ev = (P * np.arange(1, 6)).sum(-1)
        err = np.nanmean(ev - self.Y, axis=0); order = np.argsort(-np.abs(err))[:k]
        return "\n".join(f"item {self.tgt[j]} '{TEXT_OF_ITEM[self.tgt[j]]}': model too {'high' if err[j] > 0 else 'low'} "
                         f"by {abs(err[j]):.2f} (mean predicted {np.nanmean(ev[:, j]):.2f}, human {np.nanmean(self.Y[:, j]):.2f})"
                         for j in order)

    def last_descriptor(self):
        c = self._cache[self._last_key]; P = c["P"]; mu = (P * np.arange(1, 6)).sum(-1)
        sd = np.sqrt((P * (np.arange(1, 6) - mu[..., None]) ** 2).sum(-1)).mean()
        return {"prompt_length_chars": len(self._last_key), "predicted_sd": float(sd)}

    def last_objectives(self):
        c = self._cache[self._last_key]; mu = (c["P"] * np.arange(1, 6)).sum(-1)
        vr = float(np.var(mu, axis=0).mean() / self.human_between_var)
        return {"rps_cal": float(c["cal"]), "neg_vr_between": -vr}
```

- [ ] **Step 4: Run** → 1 passed (all asserts). **Step 5: Commit**

```bash
git add src/optim/fitness.py tests/optim/test_fitness.py
git commit -m "optim: calibrated-RPS fitness with cross-fitted alpha, isolation from cal/eval panels"
```

---

### Task 7: Results writer and preflight checks

**Files:**
- Create: `src/optim/results.py`, `src/optim/checks.py`, `tests/optim/test_results_checks.py`

**Interfaces:**
- Produces: `atomic_write_json(path, obj)`; `cell_id(arm, slug, cluster, seed) -> str`; `class CellWriter(root: Path, cell_id: str)` with `set_status(state, **kw)`, `write_config(cfg)`, `write_meta(meta)`, `append_history(rec)`, `write_best(g, rec)`, `write_front(front)`, `finish(ledger)`, `write_eval(obj)` (refuses if `eval_frozen.json` exists; moves nothing silently), `archive_attempt()` (moves `history.jsonl`, `best_genotype.json` into `attempts/<n>/`). `checks.py` CLI: `python -m src.optim.checks orientation_regression --min-corr 0.60` (re-reads every `results_experiments/evoprompt_iter2/*/*/cluster_*/after_optimization_test_answers.csv` with its `case` ids, computes the correlation of per-item model means with per-item human means before and after `flip_answers`, exits 1 unless after ≥ min-corr and after > before); `python -m src.optim.checks readout_guard --models-yaml … --tier N --n-respondents 8 --n-items 20 --out …` (for each tier model this command starts and stops a vLLM server itself, using the same command line as `cell_group.sbatch`, reads 8×20 beliefs, writes `{slug: mean_mass}`, exits 1 if any tier model is below its `gate_mass_on_scale`).

- [ ] **Step 1: Write the failing test**

```python
# tests/optim/test_results_checks.py
import json, pytest
from src.optim.results import CellWriter, cell_id
from src.optim.budget import BudgetLedger

def test_cell_writer_roundtrip(tmp_path):
    cid = cell_id("ga", "Qwen_Qwen3p6-35B-A3B", 0, 1); assert cid == "ga__Qwen_Qwen3p6-35B-A3B__c0__s1"
    w = CellWriter(tmp_path, cid); w.set_status("running", attempt=1)
    w.append_history({"eval_idx": 0}); w.append_history({"eval_idx": 1}); w.write_best({"x": 1}, {"eval_idx": 1})
    w.finish(BudgetLedger(B=2))
    d = tmp_path / cid
    assert len((d / "history.jsonl").read_text().splitlines()) == 2
    assert json.loads((d / "status.json").read_text())["state"] == "running"
    w.write_eval({"a": 1})
    with pytest.raises(FileExistsError): w.write_eval({"a": 2})

def test_archive_attempt(tmp_path):
    w = CellWriter(tmp_path, "x"); w.append_history({"eval_idx": 0}); w.archive_attempt()
    assert (tmp_path / "x/attempts/1/history.jsonl").exists() and not (tmp_path / "x/history.jsonl").exists()
```

- [ ] **Step 2: Run** → FAIL

- [ ] **Step 3: Implement `results.py`**

```python
# src/optim/results.py
import json, os, shutil
from datetime import datetime, timezone
from pathlib import Path

def atomic_write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float)); os.replace(tmp, path)

def cell_id(arm, slug, cluster, seed): return f"{arm}__{slug}__c{cluster}__s{seed}"

def now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")

class CellWriter:
    def __init__(self, root, cid):
        self.dir = Path(root) / cid; self.dir.mkdir(parents=True, exist_ok=True); self.cid = cid
    def set_status(self, state, **kw): atomic_write_json(self.dir / "status.json", {"state": state, "updated": now(), **kw})
    def write_config(self, cfg): atomic_write_json(self.dir / "config.json", cfg)
    def write_meta(self, meta): atomic_write_json(self.dir / "run_meta.json", meta)
    def append_history(self, rec):
        with open(self.dir / "history.jsonl", "a") as f: f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n"); f.flush(); os.fsync(f.fileno())
    def write_best(self, g, rec): atomic_write_json(self.dir / "best_genotype.json", {"genotype": g, "record": rec})
    def write_front(self, front): atomic_write_json(self.dir / "pareto_front.json", front)
    def finish(self, ledger): atomic_write_json(self.dir / "budget.json", ledger.to_dict())
    def write_eval(self, obj):
        p = self.dir / "eval_frozen.json"
        if p.exists(): raise FileExistsError(f"{p} already written; archive the attempt first")
        atomic_write_json(p, obj)
    def archive_attempt(self):
        n = 1
        while (self.dir / "attempts" / str(n)).exists(): n += 1
        dst = self.dir / "attempts" / str(n); dst.mkdir(parents=True)
        for name in ("history.jsonl", "best_genotype.json", "budget.json", "pareto_front.json", "eval_frozen.json", "run_meta.json"):
            if (self.dir / name).exists(): shutil.move(str(self.dir / name), dst / name)
        return dst
```

- [ ] **Step 4: Implement `checks.py`**

```python
# src/optim/checks.py
import argparse, glob, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd, yaml
from src.optim.orientation import flip_answers
from src.optim.panels import load_corpus, ITEM_SPLIT, persona_scores, answers
from src.optim.results import atomic_write_json

ROOT = Path(__file__).resolve().parents[2]
IDS = list(range(1, 121))

def orientation_regression(min_corr):
    df = load_corpus().set_index("case"); before, after = [], []
    for f in glob.glob(str(ROOT / "results_experiments/evoprompt_iter2/*/*/cluster_*/after_optimization_test_answers.csv")):
        m = pd.read_csv(f).dropna()
        if m.empty: continue
        X = m[[f"i{i}" for i in IDS]].to_numpy(float); H = df.loc[m["case"], [f"i{i}" for i in IDS]].to_numpy(float)
        hm = np.nanmean(H, 0)
        before.append(np.corrcoef(np.nanmean(X, 0), hm)[0, 1]); after.append(np.corrcoef(np.nanmean(flip_answers(X, IDS), 0), hm)[0, 1])
    b, a = float(np.mean(before)), float(np.mean(after))
    print(f"runs={len(before)} per-item mean corr before={b:.3f} after={a:.3f}")
    return 0 if (a >= min_corr and a > b) else 1

def _serve(hf_id, tp, dtype, util, port, log):
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", hf_id, "--port", str(port),
           "--trust-remote-code", "--tensor-parallel-size", str(tp), "--max-model-len", "8192",
           "--gpu-memory-utilization", str(util), "--enable-prefix-caching", "--max-logprobs", "20"]
    if dtype == "fp8": cmd += ["--quantization", "fp8"]
    return subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)

def readout_guard(models_yaml, tier, n_resp, n_items, out):
    import urllib.request
    from src.optim.readout import VLLMReadout
    from src.optim.persona import system_prompt
    from src.optim.genotype import seed_genotype
    cfg = yaml.safe_load(open(models_yaml)); d = cfg["defaults"]; res = {}; bad = []
    df = load_corpus(); rows = df[df["clusters"] == 0].head(n_resp)
    systems = [system_prompt(seed_genotype(0), persona_scores(x)) for x in answers(rows, ITEM_SPLIT[0])]
    for m in cfg["models"]:
        if m.get("tier") != tier or m.get("readout", d["readout"]) != "belief_vllm": continue
        port = 8000 + (abs(hash(m["hf_id"])) % 1000)
        proc = _serve(m["hf_id"], m.get("tp", d["tp"]), m.get("dtype", d["dtype"]), m.get("gpu_memory_utilization", d["gpu_memory_utilization"]),
                      port, f"/tmp/guard_{m['slug']}.log")
        try:
            for _ in range(180):
                try: urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5); break
                except Exception:
                    if proc.poll() is not None: break
                    time.sleep(10)
            if proc.poll() is not None: res[m["slug"]] = {"mass": None, "error": "server exited"}; bad.append(m["slug"]); continue
            r = VLLMReadout(m["hf_id"], f"http://127.0.0.1:{port}/v1", m.get("prefill", d["prefill"]), workers=32)
            _, M, _ = r.beliefs(systems, ITEM_SPLIT[1][:n_items]); mass = float(M.mean())
            res[m["slug"]] = {"mass": mass, "pass": mass >= m.get("gate_mass_on_scale", d["gate_mass_on_scale"])}
            if not res[m["slug"]]["pass"]: bad.append(m["slug"])
            print(f"{m['slug']:45s} mass={mass:.3f} {'PASS' if res[m['slug']]['pass'] else 'FAIL'}", flush=True)
        finally:
            proc.terminate(); proc.wait(timeout=120)
    atomic_write_json(out, res); return 1 if bad else 0

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("orientation_regression"); o.add_argument("--min-corr", type=float, default=0.60)
    g = sub.add_parser("readout_guard"); g.add_argument("--models-yaml", required=True); g.add_argument("--tier", type=int, default=1)
    g.add_argument("--n-respondents", type=int, default=8); g.add_argument("--n-items", type=int, default=20); g.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.exit(orientation_regression(a.min_corr) if a.cmd == "orientation_regression"
             else readout_guard(a.models_yaml, a.tier, a.n_respondents, a.n_items, a.out))

if __name__ == "__main__":
    main()
```

The preflight job requests one GPU; `readout_guard` serves models one at a time on it, so a `tp: 4` model (Qwen3-235B-FP8) must be guarded in a separate `sbatch --gres=gpu:4 icml2027/slurm/preflight.sbatch 1` run, or skipped with a logged reason until its wave.

- [ ] **Step 5: Run** `python -m pytest tests/optim -q` → all passed. Then locally (CPU): `python -m src.optim.checks orientation_regression --min-corr 0.60` → prints `after ≥ 0.60`, exit 0. If `after < 0.60`, stop: the key or the artefacts disagree with the audit's +0.63 and that must be resolved before anything else.

- [ ] **Step 6: Commit**

```bash
git add src/optim/results.py src/optim/checks.py tests/optim/test_results_checks.py
git commit -m "optim: atomic cell writer, orientation regression and readout guard checks"
```

---

## Self-review

- Spec coverage: P0.1 orientation (Tasks 1, 7) ✓; P0.2 calibrated RPS fitness with cross-fitting (Tasks 2, 6) ✓, `S₀` ablation via `objective="s0"` ✓; P0.3 disjoint input/target items (Task 4) ✓; P0.4 budget accounting (Task 3) ✓; P0.5 panels (Task 4) ✓; results contract (Task 7) ✓; readout guard (Task 7) ✓. P0.6 pre-registration and P0.7 source recovery are not code; they are gates in `icml2027/STATUS.md`.
- Names used by plan 2 match: `FitnessResult` fields, `Fitness.{evaluate, evaluate_subset, per_respondent_scores, feedback, last_descriptor, last_objectives, n_opt, forward_passes_per_candidate, last_prompt_tokens}`, `BudgetLedger.{charge, charge_fraction, used, remaining, remaining_fraction, to_dict}`, `CellWriter.{append_history, write_best, write_front, finish}`, `parse_genotype, GenotypeParseError, to_json`.
- Known simplification to verify in E0: persona facet scores are computed from two input items per facet; the audit's personas used the same construction (App. "disjoint"), so results remain comparable.
