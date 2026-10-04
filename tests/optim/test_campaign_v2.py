import numpy as np
import pandas as pd
import pytest

from aamas2027.pipeline import global_split, neutral_genotype, fingerprint_exposures, audit_split


def corpus(n=30):
    d = {'case': np.arange(n), 'clusters': np.arange(n) % 4}
    d.update({f'i{i}': ((np.arange(n) + i) % 5) + 1 for i in range(1, 121)})
    return pd.DataFrame(d)


def test_split_invariant_to_cluster_labels_and_disjoint():
    df = corpus()
    sizes = dict(train=5, opt=4, select=3, cal=3, eval=4)
    a = global_split(df, sizes, 17, {1, 2, 3})
    df['clusters'] = 999
    assert a == global_split(df, sizes, 17, {1, 2, 3})
    assert len(set(sum(a.values(), []))) == 19
    assert not set(sum(a.values(), [])) & {1, 2, 3}
    audit_split(a, {1, 2, 3})
    with pytest.raises(ValueError):
        audit_split({**a, 'eval': a['train']}, set())


def test_missing_input_and_duplicate_ids_fail_closed():
    df = corpus()
    df.loc[0, 'i1'] = np.nan
    assert 0 not in sum(global_split(df, dict(train=10, opt=4, select=3, cal=3, eval=9), 2, set()).values(), [])
    df.loc[1, 'case'] = df.loc[2, 'case']
    with pytest.raises(ValueError, match='duplicate'):
        global_split(df, dict(train=1, opt=2), 2, set())


def test_fingerprint_excludes_all_ambiguous_matches_and_rejects_unknown():
    df = corpus()
    ids, receipt = fingerprint_exposures(df, df[['i61', 'i62']].iloc[[0]].to_numpy(), [61, 62])
    assert set(ids) == {0, 5, 10, 15, 20, 25}
    assert receipt['ambiguous_rows'] == 1
    with pytest.raises(ValueError, match='unmatched'):
        fingerprint_exposures(df, np.array([[5, 5]]), [61, 62])


def test_neutral_seed_does_not_read_cluster_prompt(monkeypatch):
    from src.optim import genotype
    monkeypatch.setattr(genotype, 'seed_genotype', lambda _: pytest.fail('oracle seed'))
    g = neutral_genotype()
    assert len(g['trait_formulations']) == 5
    assert len(g['facet_formulations']) == 30


def test_shortlist_requires_full_panel_scores():
    from aamas2027.runner import candidates
    g = neutral_genotype()
    bad = {**g, 'role_definition': 'minibatch lucky winner'}
    h = [dict(genotype=g, rps_cal_opt=.2, full_opt_panel=True),
         dict(genotype=bad, rps_cal_opt=.01, full_opt_panel=False)]
    assert candidates(h, g, 5) == [g]
    with pytest.raises(ValueError, match='full-panel'):
        candidates(h, bad, 5)
