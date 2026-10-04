"""Same-history, target-supervised CPU references for the fresh global panel.

Selection reads train and select only. Evaluation refits the frozen specification
on train and writes one immutable set of paired respondent predictions/losses.
The curve measures retained predictive information, not questionnaire savings.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from aamas2027.pipeline import canonical_hash, sha256
from src.optim import scoring as S
from src.optim.panels import ITEM_SPLIT, answers, load_corpus


GRID = (0, 1, 2, 5, 10, 20, 30, 40, 60)
RIDGE_ALPHAS = (1., 10., 100., 1000.)
TREE_LEAVES = (10, 30)
ORDER_SEEDS = (20261004, 20261005, 20261006)
VERSION = 'hbs-decoder-v1'


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=1, allow_nan=False)


def validate_answers(x, *, allow_missing=True):
    x = np.asarray(x, float)
    if x.ndim != 2:
        raise ValueError('answers must be respondent by item')
    good = np.isfinite(x) & (x >= 1) & (x <= 5) & (x == np.floor(x))
    if np.any(~good & ~np.isnan(x)) or (not allow_missing and not good.all()):
        raise ValueError('answers outside 1..5 or unexpected missing answers')
    return x


def thresholds(y):
    """Make ordinal targets, preserving missing labels before comparison."""
    y = validate_answers(y)
    return np.where(np.isnan(y)[..., None], np.nan,
                    y[..., None] <= np.arange(1, 5)).astype(float).reshape(len(y), -1)


def onehot(x):
    x = validate_answers(x)
    v = np.where(np.isnan(x), 0, x).astype(int)
    return (v[..., None] == np.arange(1, 6)).reshape(len(x), -1).astype(np.float32)


def _cdf_projection(raw, n_items):
    return S.iso_project(np.asarray(raw, float).reshape(len(raw), n_items, 4))


def fit_ridge(x, y, alpha):
    """Multioutput ridge; rare missing training targets use observed rows per output."""
    features = onehot(x)
    labels = thresholds(y)
    if np.isfinite(labels).all():
        return Ridge(alpha=alpha, solver='cholesky').fit(features, labels)
    models = []
    for j in range(labels.shape[1]):
        seen = np.isfinite(labels[:, j])
        if not seen.any():
            raise ValueError(f'target threshold {j} has no observed training labels')
        models.append(Ridge(alpha=alpha, solver='cholesky').fit(features[seen], labels[seen, j]))
    return models


def ridge_cdf(model, x, n_targets):
    features = onehot(x)
    raw = (np.column_stack([m.predict(features) for m in model]) if isinstance(model, list)
           else model.predict(features))
    return _cdf_projection(raw, n_targets)


def fit_tree(x, y, leaf):
    labels = thresholds(y)
    if not np.isfinite(labels).all():
        raise ValueError('tree reference requires complete-case training targets')
    model = ExtraTreesRegressor(n_estimators=64, min_samples_leaf=leaf,
                                max_features=0.8, n_jobs=1, random_state=20261004)
    return model.fit(onehot(x), labels)


def tree_cdf(model, x, n_targets):
    return _cdf_projection(model.predict(onehot(x)), n_targets)


def loss_by_person(cdf, y):
    losses = S.per_respondent(S.rps(S.cdf_to_simplex(cdf), y))
    if not np.isfinite(losses).all():
        raise ValueError('each respondent needs an observed target')
    return losses


def _score(cdf, y):
    return float(loss_by_person(cdf, y).mean())


def fit_select(train_x, train_y, select_x, select_y, *, alphas=RIDGE_ALPHAS,
               leaves=TREE_LEAVES, order_seeds=ORDER_SEEDS, grid=GRID):
    """Fit on train, tune only with select; never accept an eval array."""
    train_x, train_y = validate_answers(train_x), validate_answers(train_y)
    select_x, select_y = validate_answers(select_x), validate_answers(select_y)
    if train_x.shape[1] != 60 or select_x.shape[1] != 60 or train_y.shape[1] != 60 or select_y.shape[1] != 60:
        raise ValueError('IPIP120 requires 60 input and 60 target items')
    prior = S.pop_cdf(train_y)
    choices = {'prior': {'select_rps': _score(np.broadcast_to(prior, (len(select_y), 60, 4)), select_y)}}
    orders = [np.random.default_rng(seed).permutation(60).tolist() for seed in order_seeds]
    curves = []
    cache = {}
    for seed, order in zip(order_seeds, orders):
        entries = []
        for m in grid:
            if m == 0:
                entries.append(dict(m=0, family='prior', select_rps=choices['prior']['select_rps']))
                continue
            cols = tuple(sorted(order[:m]))
            if cols not in cache:
                candidates = []
                for alpha in alphas:
                    model = fit_ridge(train_x[:, cols], train_y, alpha)
                    score = _score(ridge_cdf(model, select_x[:, cols], 60), select_y)
                    candidates.append((score, alpha))
                cache[cols] = min(candidates)
            score, alpha = cache[cols]
            entries.append(dict(m=m, family='ridge', alpha=alpha, select_rps=score))
        curves.append(dict(seed=seed, order=order, choices=entries))
    full = curves[0]['choices'][-1]
    choices['ridge'] = dict(alpha=full['alpha'], select_rps=full['select_rps'])
    tree_candidates = []
    for leaf in leaves:
        model = fit_tree(train_x, train_y, leaf)
        tree_candidates.append((_score(tree_cdf(model, select_x, 60), select_y), leaf))
    score, leaf = min(tree_candidates)
    choices['extra_trees'] = dict(leaf=leaf, select_rps=score)
    winner = min(choices, key=lambda name: (choices[name]['select_rps'], name))
    return dict(version=VERSION, grid=list(grid), curves=curves, choices=choices,
                selected_family=winner, selected=choices[winner],
                selection_rule='minimum respondent-mean raw RPS on select; tie by family/parameter',
                train_only_fitting=True, eval_used_for_selection=False)


def predict_frozen(frozen, train_x, train_y, eval_x, eval_y):
    """Return full-history references and all predeclared nested ridge curves."""
    prior = S.pop_cdf(train_y)
    forecasts = {'prior': np.broadcast_to(prior, (len(eval_y), 60, 4)).copy()}
    a = frozen['choices']['ridge']['alpha']
    forecasts['ridge'] = ridge_cdf(fit_ridge(train_x, train_y, a), eval_x, 60)
    leaf = frozen['choices']['extra_trees']['leaf']
    forecasts['extra_trees'] = tree_cdf(fit_tree(train_x, train_y, leaf), eval_x, 60)
    curves = []
    cache = {}
    for curve in frozen['curves']:
        order = curve['order']
        per = []
        for choice in curve['choices']:
            m = choice['m']
            if m == 0:
                pred = forecasts['prior']
            else:
                cols = tuple(sorted(order[:m]))
                key = (cols, choice['alpha'])
                if key not in cache:
                    cache[key] = ridge_cdf(fit_ridge(train_x[:, cols], train_y, choice['alpha']),
                                            eval_x[:, cols], 60)
                pred = cache[key]
            per.append(loss_by_person(pred, eval_y).tolist())
        curves.append(dict(seed=curve['seed'], m=frozen['grid'], per_respondent=per,
                           mean_rps=np.mean(per, axis=1).tolist()))
    losses = {name: loss_by_person(pred, eval_y).tolist() for name, pred in forecasts.items()}
    return forecasts, dict(per_respondent=losses, mean_rps={k: float(np.mean(v)) for k, v in losses.items()},
                           selected_family=frozen['selected_family'], curves=curves,
                           n_respondents=len(eval_y), n_target_items=60,
                           interpretation='retained information under the registered decoder; not literal saved answers')


def mstar_summary(curves, target_per_respondent, *, B=2000, seed=20261004):
    """Paired uncertainty for a later frozen simulator's per-person RPS.

    Resample respondents and the three predeclared input orders; return censoring
    frequencies instead of replacing out-of-grid crossings with arbitrary values.
    The caller must supply scores from the identical ordered evaluation people.
    """
    from aamas2027.analysis import retained_information

    grid = np.asarray(curves[0]['m'], float)
    losses = np.asarray([c['per_respondent'] for c in curves], float)
    target = np.asarray(target_per_respondent, float)
    if (losses.ndim != 3 or losses.shape[1] != len(grid) or
            losses.shape[2] != len(target) or not np.isfinite(losses).all() or
            not np.isfinite(target).all() or B < 1):
        raise ValueError('finite paired order-by-budget-by-person losses required')
    point = retained_information(grid, losses.mean((0, 2)), target.mean())
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        people = rng.integers(len(target), size=len(target))
        orders = rng.integers(len(curves), size=len(curves))
        draws.append(retained_information(grid, losses[orders][:, :, people].mean((0, 2)),
                                          target[people].mean()))
    statuses = {status: sum(d['status'] == status for d in draws) / B
                for status in ('interpolated', 'below_grid', 'above_grid')}
    finite = [d['m_star'] for d in draws if d['status'] == 'interpolated']
    return dict(point=point, bootstrap_status_fraction=statuses,
                conditional_interpolated_ci95=(np.quantile(finite, [.025, .975]).tolist() if finite else None),
                bootstrap_draws=B, resampling='paired respondents and predeclared input orders',
                interpretation='retained information under the registered decoder; not literal saved answers')


def _verified_inputs(old_root, results, manifest):
    old_root, results, manifest = Path(old_root), Path(results), Path(manifest)
    man = json.loads(manifest.read_text())
    panels = json.loads((results / 'panels.json').read_text())
    if canonical_hash({k: v for k, v in man.items() if k != 'sha256'}) != man['sha256']:
        raise ValueError('manifest hash mismatch')
    if canonical_hash({k: v for k, v in panels.items() if k != 'sha256'}) != panels['sha256']:
        raise ValueError('panel hash mismatch')
    corpus = old_root / 'data/raw/df_ipipneo_120_clusters'
    if sha256(corpus) != panels['source_sha256']:
        raise ValueError('corpus hash mismatch')
    expected = {'train': 8000, 'opt': 80, 'select': 120, 'cal': 120, 'eval': 600}
    if man['panel_sizes'] != expected or any(len(panels['splits'][k]) != n for k, n in expected.items()):
        raise ValueError('unexpected global panel sizes')
    if len(set(sum(panels['splits'].values(), []))) != sum(expected.values()):
        raise ValueError('respondents overlap across panels')
    return man, panels, corpus


def run(old_root, results, manifest, stage='all'):
    results = Path(results)
    man, panels, corpus = _verified_inputs(old_root, results, manifest)
    out = results / 'decoders'
    freeze_path, final_path = out / 'FROZEN.json', out / 'FINAL.json'
    if stage in ('select', 'all') and freeze_path.exists():
        raise FileExistsError(freeze_path)
    if stage in ('eval', 'all') and final_path.exists():
        raise FileExistsError(final_path)
    df = load_corpus(corpus).set_index('case', drop=False)
    inp, tgt = ITEM_SPLIT
    def panel(name):
        rows = df.loc[panels['splits'][name]]
        return answers(rows, inp), answers(rows, tgt)
    train_x, train_y = panel('train')
    if stage in ('select', 'all'):
        select_x, select_y = panel('select')
        spec = fit_select(train_x, train_y, select_x, select_y)
        spec.update(manifest_sha256=man['sha256'], panel_sha256=panels['sha256'],
                    source_sha256=panels['source_sha256'], n_train=len(train_x), n_select=len(select_x))
        spec['sha256'] = canonical_hash(spec)
        _write_new(freeze_path, spec)
    if stage in ('eval', 'all'):
        spec = json.loads(freeze_path.read_text())
        if (canonical_hash({k: v for k, v in spec.items() if k != 'sha256'}) != spec['sha256'] or
                spec['manifest_sha256'] != man['sha256'] or spec['panel_sha256'] != panels['sha256']):
            raise ValueError('decoder freeze receipt mismatch')
        eval_x, eval_y = panel('eval')
        forecasts, report = predict_frozen(spec, train_x, train_y, eval_x, eval_y)
        out.mkdir(parents=True, exist_ok=True)
        prob_path = out / 'eval_probabilities.npz'
        if prob_path.exists():
            raise FileExistsError(prob_path)
        np.savez_compressed(prob_path, **{k: S.cdf_to_simplex(v) for k, v in forecasts.items()})
        report.update(version=VERSION, frozen_sha256=spec['sha256'], panel_sha256=panels['sha256'],
                      probabilities_sha256=sha256(prob_path), eval_case_ids=panels['splits']['eval'])
        _write_new(final_path, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-root', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--stage', choices=['select', 'eval', 'all'], default='all')
    args = parser.parse_args()
    run(args.old_root, args.results, args.manifest, args.stage)
