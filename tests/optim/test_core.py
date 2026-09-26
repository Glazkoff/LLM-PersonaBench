import json
import sys

import numpy as np
import pytest

from src.optim import scoring as S
from src.optim.budget import BudgetExhausted, BudgetLedger
from src.optim.genotype import GenotypeParseError, parse_genotype, seed_genotype
from src.optim.orientation import REVERSE_MASK_120, flip_answers, flip_simplex
from src.optim.panels import FACET_OF_ITEM, ITEM_SPLIT, make_panels, persona_scores
from src.optim.persona import system_prompt
from tests.optim.conftest import TEMPLATE


def test_orientation_mask_and_flip():
    assert int(REVERSE_MASK_120.sum()) == 55
    rev = int(np.where(REVERSE_MASK_120)[0][0]) + 1
    fwd = int(np.where(~REVERSE_MASK_120)[0][0]) + 1
    P = np.zeros((1, 2, 5)); P[0, :, 0] = 1
    out = flip_simplex(P, [fwd, rev])
    assert out[0, 0, 0] == 1 and out[0, 1, 4] == 1
    X = np.array([[1., 5.]])
    assert np.array_equal(flip_answers(flip_answers(X, [fwd, rev]), [fwd, rev]), X)


def test_rps_matches_audit():
    sys.path.insert(0, "arr2026/scripts/hyp")
    import _scoring as A
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(5), size=(7, 3)); Y = rng.integers(1, 6, size=(7, 3)).astype(float)
    assert np.allclose(np.nanmean(S.rps(P, Y), axis=1), A.score("rps", P, Y))


def test_truth_beats_point_under_rps_not_under_s0():
    p = np.array([.1, .2, .4, .2, .1]); rng = np.random.default_rng(1)
    Y = rng.choice(np.arange(1, 6), p=p, size=(40000, 1)).astype(float)
    truth = np.broadcast_to(p, (40000, 1, 5)); point = np.zeros_like(truth); point[..., 2] = 1
    assert np.nanmean(S.rps(truth, Y)) < np.nanmean(S.rps(point, Y))
    assert np.nanmean(S.s0(point, Y)) < np.nanmean(S.s0(truth, Y))  # the audited metric prefers the constant


def test_iso_projection_exact():
    F = np.array([[[0.5, 0.3, 0.8, 1.2]], [[0.9, 0.1, 0.1, 0.0]]])
    out = S.iso_project(F)
    assert np.all(np.diff(out, axis=-1) >= -1e-12) and out.min() >= 0 and out.max() <= 1
    assert np.allclose(out[0, 0], [0.4, 0.4, 0.8, 1.0])
    assert np.allclose(out[1, 0], [0.275] * 4)


def test_alpha_zero_is_prior_and_crossfit_helps_informative_Q():
    rng = np.random.default_rng(3); n, J = 80, 10
    Y = rng.integers(1, 6, size=(n, J)).astype(float)
    P = np.full((n, J, 5), 0.1); P[np.arange(n)[:, None], np.arange(J)[None, :], (Y - 1).astype(int)] = 0.6
    F0 = S.pop_cdf(Y)
    assert np.allclose(S.calibrate(S.cdf(P), F0, 0.0), np.broadcast_to(S.iso_project(F0[None]), (n, J, 4)))
    score, (a1, a2) = S.crossfit_rps_cal(S.cdf(P), F0, Y, [0, .5, 1, 1.5, 2], rng)
    prior = float(np.nanmean(S.rps(S.cdf_to_simplex(np.broadcast_to(F0, (n, J, 4))), Y)))
    assert score < prior and a1 > 0 and a2 > 0


def test_genotype_parse_and_budget():
    g = json.loads(json.dumps({k: TEMPLATE[k] for k in ("role_definition", "trait_formulations",
                                                          "facet_formulations", "critic_formulations")}))
    del g["trait_formulations"]["openness"]; g["facet_formulations"]["facet_made_up"] = "x"
    out = parse_genotype("Sure!\n```json\n" + json.dumps(g) + "\n```", TEMPLATE)
    assert out["trait_formulations"]["openness"] == TEMPLATE["trait_formulations"]["openness"]
    assert "facet_made_up" not in out["facet_formulations"] and "intensity_modifiers" in out
    with pytest.raises(GenotypeParseError):
        parse_genotype("no json", TEMPLATE)
    L = BudgetLedger(3); L.charge(10); L.charge_fraction(0.2); L.charge_fraction(0.2)
    assert L.used == 2 and L.remaining == 1 and abs(L.remaining_fraction() - 1.6) < 1e-9
    L.charge(1)
    with pytest.raises(BudgetExhausted):
        L.charge(1)


def test_item_split_and_panels(corpus):
    inp, tgt = ITEM_SPLIT
    assert len(inp) == 60 and len(tgt) == 60 and not set(inp) & set(tgt)
    from collections import Counter
    assert set(Counter(FACET_OF_ITEM[i] for i in inp).values()) == {2}
    sizes = {"train": 500, "opt": 40, "cal": 40, "eval": 100}
    a = make_panels(corpus, 1, 7, sizes, exclude={10_000_000})
    b = make_panels(corpus, 1, 7, sizes, exclude={10_000_000})
    ids = [set(p["case"]) for p in (a.train, a.opt, a.cal, a.eval)]
    assert sum(len(s) for s in ids) == len(set().union(*ids)) and 10_000_000 not in set().union(*ids)
    assert list(a.eval["case"]) == list(b.eval["case"]) and (a.eval["clusters"] == 1).all()


def test_persona_uses_every_facet_and_differs_by_person():
    for c in range(4):
        g = seed_genotype(c)
        s = system_prompt(g, persona_scores(np.full(60, 1.0)))
        assert s.count("This facet") == len(g["facet_formulations"]), f"facet name mismatch in cluster {c}"
        assert s.count("This trait") == 5
    g = seed_genotype(0)
    assert system_prompt(g, persona_scores(np.full(60, 1.0))) != system_prompt(g, persona_scores(np.full(60, 5.0)))
