import json

import numpy as np
import pytest

from src.optim.arms import ARMS
from src.optim.arms.qd import crowding_distance, knee_point, nondominated_sort
from src.optim.arms.reflective import pareto_front
from src.optim.budget import BudgetLedger
from src.optim.fitness import Fitness
from src.optim.panels import make_panels
from src.optim.results import CellWriter
from src.optim.run_cell import run_one
from tests.optim.conftest import TEMPLATE, JitterMutator, PersonaReadout

SIZES = {"train": 500, "opt": 40, "cal": 40, "eval": 100}
CFG = {"ga": {"population_size": 4, "generations": 3, "elites": 1, "tournament_k": 2, "crossover_rate": 0.8, "mutation_rate": 0.5},
       "de": {"population_size": 4, "generations": 3, "crossover_rate": 0.8},
       "promptbreeder": {"population_size": 4, "generations": 3, "mutation_prompt_pool": 4, "hyper_mutation_rate": 1.0},
       "opro": {"proposals_per_round": 3, "rounds": 4, "history_shown": 4},
       "protegi": {"beam": 2, "rounds": 3, "gradients_per_candidate": 1, "edits_per_gradient": 1},
       "gepa": {"minibatch_respondents": 8, "pareto_pool_max": 6, "reflection_items_shown": 5},
       "mapelites": {"islands": 2, "init_per_island": 2, "iterations": 8, "migrate_every": 3, "insight_window": 3,
                     "archive_bins": {"prompt_length_chars": [0, 1500, 2500, 4000, 100000], "predicted_sd": [0, .8, 1, 1.2, 9]}},
       "nsga2": {"population_size": 4, "generations": 3, "tournament_k": 2, "crossover_rate": 0.8, "mutation_rate": 0.5},
       "random": {}, "paraphrase": {}, "bestofb": {}}


@pytest.mark.parametrize("arm", sorted(CFG))
def test_every_arm_exhausts_budget_exactly(arm, corpus, tmp_path):
    p = make_panels(corpus, 0, 1, SIZES)
    fit = Fitness(p, PersonaReadout(), [0, .5, 1], np.random.default_rng(0))
    w = CellWriter(tmp_path, arm)
    led = BudgetLedger(12)
    a = ARMS[arm](CFG[arm], np.random.default_rng(1), JitterMutator(fail_every=5), TEMPLATE)
    best = a.run(fit, TEMPLATE, led, w)
    assert abs(led.used_f - 12) < 1e-9, (arm, led.used_f)
    assert best is not None and "role_definition" in best
    hist = [json.loads(l) for l in (tmp_path / arm / "history.jsonl").read_text().splitlines()]
    assert hist[0]["operation"] == "init"
    assert any(not h["parse_ok"] for h in hist) or arm in ("gepa",)
    if arm not in ("paraphrase", "nsga2"):
        assert fit.evaluate(best).rps_cal_opt <= hist[0]["rps_cal_opt"] + 1e-9


def test_fitness_never_reads_cal_or_eval(corpus):
    p = make_panels(corpus, 0, 1, SIZES)

    class Trip:
        def __getattr__(self, k):
            raise AssertionError("cal/eval touched")

        def __getitem__(self, k):
            raise AssertionError("cal/eval touched")
    p.cal, p.eval = Trip(), Trip()
    f = Fitness(p, PersonaReadout(), [0, .5, 1], np.random.default_rng(2))
    r = f.evaluate(TEMPLATE)
    assert 0 < r.rps_cal_opt < 1 and f.per_respondent_scores(TEMPLATE).shape == (40,)
    assert "model too" in f.feedback(TEMPLATE, 3) and set(f.last_objectives()) == {"rps_cal", "neg_vr_between"}


def test_moo_helpers():
    F = np.array([[1, 5], [2, 3], [3, 1], [4, 4], [5, 5]], float)
    assert nondominated_sort(F)[:3] == [[0, 1, 2], [3], [4]]
    d = crowding_distance(F, [0, 1, 2]); assert np.isinf(d[0]) and np.isfinite(d[1])
    assert knee_point(np.array([[0, 1], [.2, .2], [1, 0]], float), [0, 1, 2]) == 1
    fr, w = pareto_front({"a": np.array([.1, .5, .5]), "b": np.array([.5, .1, .5]), "c": np.array([.6, .6, .6])})
    assert set(fr) == {"a", "b"}


def test_run_one_end_to_end_and_idempotent(corpus, tmp_path):
    cell = {"cell_id": "gepa__fake__c0__s1", "arm": "gepa", "fitness": "rps_cal", "model_slug": "fake", "hf_id": "fake",
            "cluster": 0, "seed": 1, "budget_B": 6, "arm_cfg": CFG["gepa"], "panel_sizes": SIZES,
            "alphas": [0, .5, 1], "exclude_globs": [], "boot_B": 100}
    st = run_one(cell, corpus, PersonaReadout(), JitterMutator(), tmp_path, {"git_sha": "x"})
    d = tmp_path / cell["cell_id"]
    assert st == "completed", json.loads((d / "run_meta.json").read_text()).get("error")
    ev = json.loads((d / "eval_frozen.json").read_text())
    for k in ("panel", "base", "evolved", "delta", "floors"):
        assert k in ev
    assert "wrong_persona_rps_cal" in ev["evolved"] and (d / "belief_probs.npz").exists()
    assert len(json.loads((d / "per_respondent.json").read_text())["base_rps_cal"]) == 100
    assert run_one(cell, corpus, PersonaReadout(), JitterMutator(), tmp_path, {}) == "completed"


def test_s0_ablation_runs(corpus, tmp_path):
    cell = {"cell_id": "ga-s0__fake__c0__s1", "arm": "ga", "fitness": "s0", "model_slug": "fake", "hf_id": "fake",
            "cluster": 0, "seed": 1, "budget_B": 6, "arm_cfg": CFG["ga"], "panel_sizes": SIZES,
            "alphas": [0, .5, 1], "exclude_globs": [], "boot_B": 50}
    assert run_one(cell, corpus, PersonaReadout(), JitterMutator(), tmp_path, {}) == "completed"
    h = json.loads((tmp_path / cell["cell_id"] / "history.jsonl").read_text().splitlines()[0])
    assert h["rps_cal_opt"] == h["s0_opt"] and h["rps_cal_true_opt"] == h["rps_cal_true_opt"]


def test_cache_keeps_reevaluated_genotype(corpus):
    """Regression: re-evaluating a genotype already in a full cache must not evict the fresh entry."""
    import copy
    p = make_panels(corpus, 0, 1, SIZES)
    f = Fitness(p, PersonaReadout(), [0, .5, 1], np.random.default_rng(0))
    gs = []
    for i in range(40):
        g = copy.deepcopy(TEMPLATE); g["role_definition"] += f" v{i}"; gs.append(g)
        f.evaluate(g)
        f.evaluate_subset(g, np.arange(4))
    f.evaluate(gs[5])  # identical to an old, long-evicted-or-oldest entry
    assert f.per_respondent_scores(gs[5]).shape == (40,)
    f.evaluate(gs[39]); f.evaluate(gs[39])
    assert f.per_respondent_scores(gs[39]).shape == (40,)


def test_feedback_survives_cache_eviction(corpus):
    """Regression: ProTeGi asks for feedback on beam members evaluated many candidates ago."""
    import copy
    p = make_panels(corpus, 0, 1, SIZES)
    f = Fitness(p, PersonaReadout(), [0, .5, 1], np.random.default_rng(0))
    old = copy.deepcopy(TEMPLATE); old["role_definition"] += " oldest"
    f.evaluate(old)
    for i in range(80):
        g = copy.deepcopy(TEMPLATE); g["role_definition"] += f" n{i}"
        f.evaluate(g); f.evaluate_subset(g, np.arange(4))
    assert "model too" in f.feedback(old, 3)
    assert f.per_respondent_scores(old).shape == (40,)
