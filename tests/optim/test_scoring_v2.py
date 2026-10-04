"""Regressions for the version 2 search-time scoring estimator."""

import numpy as np
import pytest

from src.optim import scoring as S
from src.optim.fitness import Fitness
from src.optim.panels import make_panels
from tests.optim.conftest import TEMPLATE, toy_corpus


def test_bounded_isotonic_projects_raw_values_before_bounds():
    x = np.array([[1.4, .5, .9, 1.0]])
    assert np.allclose(S.iso_project(x), [[.933333333333, .933333333333, .933333333333, 1.]])
    assert np.allclose(S.iso_project(np.array([[-2., 1.5]])), [[0., 1.]])
    assert np.allclose(S.iso_project(np.array([[1.4, .5]])), [[.95, .95]])


@pytest.mark.parametrize("bad", [
    np.array([[[.2, .2, .2, .2, .1]]]),
    np.array([[[.2, .2, .2, .2, np.nan]]]),
    np.array([[[1.1, -.1, 0., 0., 0.]]]),
])
def test_rps_rejects_invalid_simplex(bad):
    with pytest.raises(ValueError):
        S.rps(bad, np.array([[1.]]))


def test_missing_targets_are_masked_and_malformed_targets_rejected():
    P = np.array([[[1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.]]])
    out = S.rps(P, np.array([[np.nan, 1.]]))
    assert np.isnan(out[0, 0]) and out[0, 1] == 0
    for bad in (0., 6., 2.5, np.inf):
        with pytest.raises(ValueError):
            S.rps(P[:, :1], np.array([[bad]]))
    with pytest.raises(ValueError):
        S.select_alpha(S.cdf(P), np.array([[.2, .4, .6, .8], [.2, .4, .6, .8]]),
                       np.full((1, 2), np.nan), [0., 1.])


def test_crossfit_centers_on_fit_half_and_returns_true_oof_cells():
    Q = np.array([[[.1, .2, .3, .4]], [[.2, .3, .4, .5]],
                  [[.8, .85, .9, .95]], [[.9, .93, .96, .99]]])
    Y = np.array([[1.], [2.], [4.], [5.]])
    F0 = np.array([[.2, .4, .6, .8]])
    folds = (np.array([0, 1]), np.array([2, 3]))
    score, alphas, cells = S.crossfit_rps_cal(Q, F0, Y, [0., 1.], np.random.default_rng(9),
                                             folds=folds, return_cells=True)
    aA, _ = S.select_alpha(Q[folds[0]], F0, Y[folds[0]], [0., 1.])
    aB, _ = S.select_alpha(Q[folds[1]], F0, Y[folds[1]], [0., 1.])
    expected = np.empty_like(Y)
    expected[folds[1]] = S.cal_cells(Q[folds[1]], F0, Y[folds[1]], aA, Q[folds[0]].mean(0))
    expected[folds[0]] = S.cal_cells(Q[folds[0]], F0, Y[folds[0]], aB, Q[folds[1]].mean(0))
    assert alphas == (aA, aB)
    assert np.allclose(cells, expected)
    assert score == pytest.approx(np.nanmean(expected))


def test_fitness_folds_stay_fixed_and_reflective_scores_are_oof():
    class Readout:
        def beliefs(self, systems, item_ids):
            n, j = len(systems), len(item_ids)
            P = np.full((n, j, 5), .1)
            for i, s in enumerate(systems):
                P[i, :, (len(s) + i) % 5] = .6
            return P, np.ones((n, j)), 0

    panels = make_panels(toy_corpus(300, seed=1), 0, 2,
                         {"train": 80, "opt": 12, "cal": 12, "eval": 20})
    f = Fitness(panels, Readout(), [0., .5, 1.], np.random.default_rng(7))
    result = f.evaluate(TEMPLATE)
    folds = tuple(x.copy() for x in f.folds)
    P = f._cache[f._last_key]["P"]
    _, _, cells = S.crossfit_rps_cal(S.cdf(P), f.F0, f.Y, f.alphas, np.random.default_rng(99),
                                    folds=folds, return_cells=True)
    assert np.allclose(f.per_respondent_scores(TEMPLATE), S.per_respondent(cells))
    assert result.rps_cal_opt == pytest.approx(np.nanmean(S.per_respondent(cells)))
    f.evaluate({**TEMPLATE, "role_definition": TEMPLATE["role_definition"] + " variant"})
    assert all(np.array_equal(a, b) for a, b in zip(f.folds, folds))
    rng_state = repr(f.rng.bit_generator.state)
    subset = np.arange(6)
    one = f.evaluate_subset(TEMPLATE, subset)
    two = f.evaluate_subset(TEMPLATE, subset)
    assert one.rps_cal_opt == two.rps_cal_opt
    assert repr(f.rng.bit_generator.state) == rng_state


def test_raw_rps_objective_uses_raw_score_and_per_person_losses():
    class Readout:
        def beliefs(self, systems, item_ids):
            P = np.full((len(systems), len(item_ids), 5), .2)
            return P, np.ones(P.shape[:2]), 0

    panels = make_panels(toy_corpus(300, seed=2), 0, 3,
                         {"train": 80, "opt": 12, "cal": 12, "eval": 20})
    f = Fitness(panels, Readout(), [0., 1.], np.random.default_rng(0), objective="rps_raw")
    result = f.evaluate(TEMPLATE)
    per = S.per_respondent(S.rps(f._cache[f._last_key]["P"], f.Y))
    assert result.rps_cal_opt == pytest.approx(np.nanmean(per))
    assert result.rps_raw_opt == pytest.approx(np.nanmean(per))
    assert np.allclose(f.per_respondent_scores(TEMPLATE), per)


def test_invalid_subset_is_rejected_before_any_forecast():
    from src.optim.fitness import Fitness
    f = object.__new__(Fitness)
    f.n_opt = 6
    f._beliefs = lambda *args: (_ for _ in ()).throw(AssertionError('forecast executed'))
    for idx in ([1, 1], [1], [-1, 2], [0, 6], [0.5, 1]):
        with pytest.raises(ValueError, match='subset'):
            f.evaluate_subset({}, idx)
