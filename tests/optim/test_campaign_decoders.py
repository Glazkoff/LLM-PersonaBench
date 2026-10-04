import numpy as np
import pytest

from aamas2027 import decoders as D


def _signal(seed=4, n=260):
    rng = np.random.default_rng(seed)
    x = rng.integers(1, 6, size=(n, 60)).astype(float)
    # Every target has a real same-person signal in the available history.
    y = x[:, np.arange(60) % 5].copy()
    return x, y


def test_missing_targets_are_masked_before_threshold_comparison():
    y = np.array([[np.nan, 1, 5]])
    got = D.thresholds(y).reshape(1, 3, 4)
    assert np.isnan(got[0, 0]).all()
    np.testing.assert_array_equal(got[0, 1], [1, 1, 1, 1])
    np.testing.assert_array_equal(got[0, 2], [0, 0, 0, 0])
    x = np.array([[1., np.nan], [5., 3.]])
    assert D.onehot(x)[0, 5:].sum() == 0
    with pytest.raises(ValueError):
        D.validate_answers(np.array([[0., 2.]]))


def test_ridge_uses_observed_labels_only_and_beats_prior_on_known_signal():
    x, y = _signal()
    tr, ev = slice(None, 200), slice(200, None)
    masked = y[tr].copy()
    masked[::5, 0] = np.nan
    model = D.fit_ridge(x[tr], masked, alpha=1.)
    pred = D.ridge_cdf(model, x[ev], 60)
    assert pred.shape == (60, 60, 4)
    assert np.isfinite(pred).all()
    assert np.all(np.diff(pred, axis=-1) >= -1e-12)
    prior = np.broadcast_to(D.S.pop_cdf(masked), pred.shape)
    assert D.loss_by_person(pred, y[ev]).mean() < D.loss_by_person(prior, y[ev]).mean() - .03


def test_selection_is_frozen_before_evaluation_and_curve_is_nested():
    x, y = _signal(n=300)
    spec = D.fit_select(x[:200], y[:200], x[200:250], y[200:250],
                        alphas=(1., 100.), leaves=(10,), order_seeds=(23, 24, 25),
                        grid=(0, 1, 2, 5, 10, 20, 30, 40, 60))
    frozen = D.canonical_hash(spec)
    assert spec['selected_family'] in {'ridge', 'extra_trees', 'prior'}
    for curve in spec['curves']:
        assert sorted(curve['order']) == list(range(60))
        assert [c['m'] for c in curve['choices']] == list(D.GRID)
        assert all(c['alpha'] in (1., 100.) for c in curve['choices'][1:])
    forecasts, report = D.predict_frozen(spec, x[:200], y[:200], x[250:], y[250:])
    assert D.canonical_hash(spec) == frozen
    assert set(forecasts) == {'prior', 'ridge', 'extra_trees'}
    assert all(len(losses) == 50 for losses in report['per_respondent'].values())
    assert len(report['curves']) == 3
    assert all(np.asarray(c['per_respondent']).shape == (9, 50) for c in report['curves'])
    # Perturbing E can alter outcomes, but it cannot feed back into selected settings.
    changed = y[250:].copy()
    changed[:] = 5
    _, altered = D.predict_frozen(spec, x[:200], y[:200], x[250:], changed)
    assert D.canonical_hash(spec) == frozen
    assert report['mean_rps'] != altered['mean_rps']


def test_mstar_keeps_out_of_grid_bootstrap_mass_explicit():
    curves = [dict(m=[0, 1, 2], per_respondent=[[.3] * 12, [.2] * 12, [.1] * 12])
              for _ in range(3)]
    middle = D.mstar_summary(curves, [.15] * 12, B=30)
    assert middle['point']['m_star'] == pytest.approx(1.5)
    assert middle['bootstrap_status_fraction']['interpolated'] == 1.
    censored = D.mstar_summary(curves, [.05] * 12, B=30)
    assert censored['point']['status'] == 'above_grid'
    assert censored['conditional_interpolated_ci95'] is None
    assert censored['bootstrap_status_fraction']['above_grid'] == 1.
