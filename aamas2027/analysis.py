"""Paired respondent inference. Seeds are repeated runs, never independent people."""
import argparse
import json
from pathlib import Path

import numpy as np

from aamas2027.pipeline import canonical_hash
from src.optim.results import atomic_write_json


def paired_inference(a, b, B=5000, seed=20261004):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 2 or not np.all(np.isfinite(a + b)):
        raise ValueError('paired finite seed-by-respondent arrays required')
    d = b - a
    rng = np.random.default_rng(seed)
    ns, n = d.shape
    boots = np.empty(B)
    # Crossed bootstrap: same people recur across all seeds and methods.
    for i in range(B):
        boots[i] = d[rng.integers(ns, size=ns)][:, rng.integers(n, size=n)].mean()
    person_delta = d.mean(0)
    delta = float(person_delta.mean())
    # Paired sign flips cluster over respondent; condition on observed seed set.
    null = np.array([(rng.choice([-1, 1], n) * person_delta).mean() for _ in range(B)])
    return dict(delta=delta, ci95=np.quantile(boots, [.025, .975]).tolist(),
                p_signflip=float((1 + (np.abs(null) >= abs(delta)).sum()) / (B + 1)),
                n_respondents=n, n_seeds=ns, per_seed_delta=d.mean(1).tolist(),
                bootstrap='crossed respondents and seeds', permutation='respondent sign flip, conditional on seeds')


def holm(pvalues):
    keys = sorted(pvalues, key=pvalues.get)
    result, running = {}, 0.
    for rank, key in enumerate(keys):
        running = max(running, min(1., (len(keys) - rank) * pvalues[key]))
        result[key] = running
    return result


def budget_trace(history, full_panel_forecasts, B=80, improvement=.001):
    rows = [r for r in history if r.get('parse_ok') and r.get('full_opt_panel', True)
            and np.isfinite(r.get('rps_raw_opt', np.nan))]
    if not rows:
        raise ValueError('no valid full-panel search observations')
    target = rows[0]['rps_raw_opt'] - improvement
    costs = np.asarray([r['cumulative_forecasts'] / full_panel_forecasts for r in rows])
    scores = np.minimum.accumulate([r['rps_raw_opt'] for r in rows])
    hit = np.flatnonzero(scores <= target)
    curve = {}
    for b in (1, 5, 10, 20, 40, 80):
        eligible = np.flatnonzero(costs <= b)
        curve[str(b)] = float(scores[eligible[-1]]) if len(eligible) else None
    return dict(threshold=float(target), threshold_rule='seed raw opt RPS minus 0.001',
                reached=bool(len(hit)), first_full_panel_equivalent=float(costs[hit[0]]) if len(hit) else None,
                capped_cost=float(costs[hit[0]]) if len(hit) else B,
                not_reached_included=True, last_observed_equivalent=float(costs[-1]), best_raw_rps_at_budget=curve)


def retained_information(m, loss, target):
    m, loss = np.asarray(m, float), np.asarray(loss, float)
    if len(m) != len(loss) or len(m) < 2 or np.any(np.diff(m) <= 0) or not np.all(np.isfinite(loss)):
        raise ValueError('valid ordered decoder grid required')
    # Monotonic regression summarizes a noisy decoder curve, not best-of-prefix
    # test selection. Curve construction/prefix order was frozen before E.
    from sklearn.isotonic import IsotonicRegression
    curve = IsotonicRegression(increasing=False).fit_transform(m, loss)
    if target < curve[-1]:
        return dict(m_star=None, status='above_grid', lower_bound=float(m[-1]))
    if target > curve[0]:
        return dict(m_star=None, status='below_grid', upper_bound=float(m[0]))
    return dict(m_star=float(np.interp(-target, -curve, m)), status='interpolated',
                interpretation='retained information under the registered decoder; not literal saved answers')


def freeze(results, manifest):
    results = Path(results)
    if (results / 'FROZEN.json').exists():
        raise FileExistsError('selection already frozen')
    man = json.loads(Path(manifest).read_text())
    panels = json.loads((results / 'panels.json').read_text())
    decoder_frozen = json.loads((results / 'decoders/FROZEN.json').read_text())
    if (decoder_frozen['manifest_sha256'] != man['sha256'] or decoder_frozen['panel_sha256'] != panels['sha256']
            or canonical_hash({k: v for k, v in decoder_frozen.items() if k != 'sha256'}) != decoder_frozen['sha256']):
        raise ValueError('decoder selection provenance mismatch')
    records = {}
    for cell in man['cells']:
        p = results / 'cells' / cell['cell_id'] / 'selected.json'
        if not p.exists():
            raise RuntimeError(f'missing selected candidate: {cell["cell_id"]}')
        s = json.loads(p.read_text())
        if s['manifest_sha256'] != man['sha256'] or s['panel_sha256'] != panels['sha256']:
            raise ValueError('candidate selection provenance mismatch')
        records[cell['cell_id']] = dict(sha256=canonical_hash(s), selection_score=s['selection_score'])
    comparators = {}
    for model in sorted({c['hf_id'] for c in man['cells']}):
        means = {}
        for arm in sorted({c['arm'] for c in man['cells']} - {man['primary_method']}):
            values = [records[c['cell_id']]['selection_score'] for c in man['cells']
                      if c['hf_id'] == model and c['arm'] == arm and c['fitness'] == 'rps_raw']
            if len(values) != len(man['seeds']):
                raise RuntimeError('incomplete selection seeds')
            means[arm] = float(np.mean(values))
        comparators[model] = min(means, key=lambda k: (means[k], k))
    atomic_write_json(results / 'FROZEN.json', dict(manifest_sha256=man['sha256'], cells=records,
                                                  panel_sha256=panels['sha256'], decoder_sha256=decoder_frozen['sha256'],
                                                  primary_method=man['primary_method'], comparators=comparators))


def analyze(results, manifest):
    results = Path(results)
    man = json.loads(Path(manifest).read_text())
    frozen = json.loads((results / 'FROZEN.json').read_text())
    decoder = json.loads((results / 'decoders/FINAL.json').read_text())
    panels = json.loads((results / 'panels.json').read_text())
    if (frozen['manifest_sha256'] != man['sha256'] or frozen['panel_sha256'] != panels['sha256']
            or decoder['frozen_sha256'] != frozen['decoder_sha256']
            or decoder['panel_sha256'] != panels['sha256'] or decoder['eval_case_ids'] != panels['splits']['eval']):
        raise ValueError('decoder and simulator respondent identities differ')
    rows, losses, costs = {}, {}, {}
    for c in man['cells']:
        p = results / 'cells' / c['cell_id'] / 'final.json'
        if not p.exists():
            raise RuntimeError(f'incomplete campaign: {c["cell_id"]}')
        row = json.loads(p.read_text())
        if (row['manifest_sha256'] != man['sha256'] or row['panel_sha256'] != panels['sha256']
                or row['selected_sha256'] != frozen['cells'][c['cell_id']]['sha256']):
            raise ValueError(f'final result provenance mismatch: {c["cell_id"]}')
        rows[c['cell_id']] = row
        directory = results / 'cells' / c['cell_id']
        history = [json.loads(line) for line in (directory / 'history.jsonl').read_text().splitlines()]
        costs[c['cell_id']] = budget_trace(history, man['panel_sizes']['opt'] * 60, c['budget_B'])
        losses.setdefault((c['hf_id'], c['arm_label']), []).append((c['seed'], row['per_respondent']['evolved']))
    comparisons = {}
    for (model, arm), vals in sorted(losses.items()):
        selected = np.asarray([v for _, v in sorted(vals)])
        baseline = np.asarray([rows[c['cell_id']]['per_respondent']['base'] for c in man['cells']
                               if c['hf_id'] == model and c['arm_label'] == arm])
        comparisons[f'{model}:{arm}:base'] = paired_inference(baseline, selected)
        decoder_loss = decoder['per_respondent'][decoder['selected_family']]
        comparisons[f'{model}:{arm}:selected_decoder'] = paired_inference(
            np.broadcast_to(decoder_loss, selected.shape), selected)
        wrong = np.asarray([rows[c['cell_id']]['per_respondent']['wrong_persona'] for c in man['cells']
                            if c['hf_id'] == model and c['arm_label'] == arm])
        comparisons[f'{model}:{arm}:wrong_persona'] = paired_inference(wrong, selected)
        if arm != man['primary_method']:
            primary = np.asarray([v for _, v in sorted(losses[(model, man['primary_method'])])])
            comparisons[f'{model}:{man["primary_method"]}:{arm}'] = paired_inference(selected, primary)
    adjusted = holm({k: v['p_signflip'] for k, v in comparisons.items()})
    for k in comparisons:
        comparisons[k]['p_holm'] = adjusted[k]
    from aamas2027.decoders import mstar_summary
    information = {}
    for (model, arm), vals in sorted(losses.items()):
        # All simulator seeds address the same people. Average seeds first;
        # m* bootstrap then propagates people/order uncertainty conditionally.
        values = np.asarray([v for _, v in sorted(vals)])
        information[f'{model}:{arm}'] = mstar_summary(decoder['curves'], values.mean(0))
    atomic_write_json(results / 'ANALYSIS.json', dict(comparisons=comparisons, frozen=frozen,
        retained_information=information, selected_decoder=decoder['selected_family'],
        search_budget_curves=costs,
        complete_cells=len(rows), expected_cells=len(man['cells']),
        claim_gate='Report nulls and all arms. Superiority requires paired CI, corrected comparisons, and strongest decoder comparison.',
        scientific_evidence=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['freeze', 'analyze'])
    p.add_argument('--results', required=True)
    p.add_argument('--manifest', required=True)
    a = p.parse_args()
    (freeze if a.stage == 'freeze' else analyze)(a.results, a.manifest)
