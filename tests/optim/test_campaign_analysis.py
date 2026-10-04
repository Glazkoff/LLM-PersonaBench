import numpy as np
import pytest
from aamas2027.analysis import paired_inference, retained_information
from aamas2027.build_manifest import build, ARMS


def test_full_grid_and_objective_controls_present():
    m = build()
    assert len(m['cells']) == 84
    for model in {c['hf_id'] for c in m['cells']}:
        for seed in m['seeds']:
            assert set(c['arm'] for c in m['cells'] if c['hf_id'] == model and c['seed'] == seed and c['fitness'] == 'rps_raw') == set(ARMS)


def test_paired_ci_and_null_do_not_treat_seeds_as_people():
    a = np.zeros((3, 40))
    b = np.full((3, 40), -.1)
    result = paired_inference(a, b, B=500, seed=1)
    assert result['n_respondents'] == 40
    assert result['delta'] == pytest.approx(-.1)
    assert result['ci95'][1] < 0
    assert result['p_signflip'] <= .01
    with pytest.raises(ValueError):
        paired_inference(a, b[:, :39])


def test_information_equivalent_is_censored_not_clipped():
    assert retained_information([0, 5, 10], [.3, .2, .15], .1)['status'] == 'above_grid'
    assert retained_information([0, 5, 10], [.3, .2, .15], .4)['status'] == 'below_grid'
    assert retained_information([0, 5, 10], [.3, .2, .15], .175)['m_star'] == pytest.approx(7.5)


def test_budget_trace_keeps_nonreachers():
    from aamas2027.analysis import budget_trace
    rows = [dict(parse_ok=True, rps_raw_opt=.2, cumulative_forecasts=10),
            dict(parse_ok=True, rps_raw_opt=.1, full_opt_panel=False, cumulative_forecasts=15),
            dict(parse_ok=True, rps_raw_opt=.1999, cumulative_forecasts=20)]
    result = budget_trace(rows, 10, B=80)
    assert result['reached'] is False
    assert result['capped_cost'] == 80
    assert result['first_full_panel_equivalent'] is None
