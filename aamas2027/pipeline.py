"""Versioned global panels; no target-derived cluster routing.

This module contains no model calls. Exposure discovery and split construction run
in a finite CPU allocation and leave hashes and an explicit coverage receipt.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.optim.panels import ITEM_COLS, ITEM_SPLIT, FACET_NAME, load_corpus, make_panels, excluded_cases
from src.optim.persona import TRAIT_OF
from src.optim.results import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def audit_split(splits, excluded):
    flat = sum(splits.values(), [])
    if len(flat) != len(set(flat)):
        raise ValueError('duplicate respondent across panels')
    if set(flat) & set(excluded):
        raise ValueError('historical exposure in new panels')


def global_split(df, sizes, seed, excluded):
    if df.case.duplicated().any():
        raise ValueError('duplicate corpus IDs')
    # Complete-case primary cohort is fixed once for every method; no arm-specific
    # deletion is allowed. Missingness generalization is a separate study.
    valid = df[ITEM_COLS].isin([1, 2, 3, 4, 5]).all(axis=1)
    pool = df.loc[valid & ~df.case.isin(excluded)].sort_values('case')
    if len(pool) < sum(sizes.values()):
        raise ValueError(f'insufficient unexposed complete cases: {len(pool)}')
    ids = pool.case.to_numpy()[np.random.default_rng(seed).permutation(len(pool))]
    out, at = {}, 0
    for name, size in sizes.items():
        out[name] = [int(i) for i in ids[at:at + size]]
        at += size
    audit_split(out, excluded)
    return out


def neutral_genotype():
    sysm = json.loads((ROOT / 'src/prompt/system.json').read_text())
    # Standard construct names only; no cluster centroids or learned prose.
    return dict(role_definition='You are a simulated respondent. Answer each statement according to the supplied personality profile.',
                trait_formulations={t: f'Tendency toward {t}.' for t in TRAIT_OF.values()},
                facet_formulations={v: f'Tendency toward {v.lower()}.' for v in FACET_NAME.values()},
                critic_formulations='Use only the supplied profile. Interpret the response scale consistently.',
                intensity_modifiers=sysm['intensity_modifiers'])


def _rowkeys(a):
    a = np.asarray(a, float)
    # Saved historical arrays use NaN and source files sometimes use zero for
    # missing. All invalid values have the same missing marker for matching.
    valid = np.isfinite(a) & (a >= 1) & (a <= 5) & (a == np.floor(a))
    x = np.ascontiguousarray(np.where(valid, a, 0), dtype=np.uint8)
    return x.view(np.dtype((np.void, x.shape[1]))).reshape(-1)


def fingerprint_exposures(df, saved, item_ids, cache=None):
    if len(item_ids) != np.asarray(saved).shape[1]:
        raise ValueError('saved exposure shape differs from item IDs')
    cache = {} if cache is None else cache
    key = tuple(item_ids)
    if key not in cache:
        keys = _rowkeys(df[[f'i{i}' for i in item_ids]].to_numpy())
        cache[key] = np.unique(keys, return_inverse=True, return_counts=True)
    unique, inverse, counts = cache[key]
    old = _rowkeys(saved)
    pos = np.searchsorted(unique, old)
    ok = pos < len(unique)
    ok[ok] &= unique[pos[ok]] == old[ok]
    if not ok.all():
        raise ValueError(f'unmatched historical responses: {int((~ok).sum())}/{len(old)}')
    matched = np.isin(inverse, np.unique(pos))
    return set(map(int, df.case[matched])), dict(rows=len(old), matched_cases=int(matched.sum()),
                                               ambiguous_rows=int((counts[pos] > 1).sum()))


def collect_exposures(df, old_root):
    """Conservative reconstruction across saved panels and *all* old manifests.
    Historical train/target answer arrays without explicit IDs are matched to the
    corpus, including every ambiguous match. Unknown shapes fail rather than
    silently declaring respondents fresh. Only IPIP120 arrays belong here.
    """
    old_root = Path(old_root)
    excluded, receipts = set(), []
    # H18 retained only per-person losses, but its script fixes the sampling
    # seed. Record this reconstruction separately (not an invented saved ID).
    for path in sorted(old_root.glob('arr2026/results*/h18*/summary.json')):
        info = json.loads(path.read_text())
        clusters = info.get('clusters', [])
        if not clusters or info['n'] % len(clusters):
            raise ValueError(f'cannot reconstruct H18 panel: {path}')
        rng = np.random.default_rng(260913)
        count = info['n'] // len(clusters)
        for c in clusters:
            pool = np.flatnonzero(df.clusters.to_numpy() == c)
            excluded.update(map(int, df.iloc[rng.choice(pool, min(count, len(pool)), replace=False)].case))
        receipts.append(dict(path=str(path), sha256=sha256(path), kind='h18_script_reconstruction', seed=260913,
                             limitation='seed from archived script default; H18 summary did not persist it'))
    for path in sorted(old_root.glob('arr2026/results*/**/*case_ids.csv')):
        import pandas as pd
        saved = pd.read_csv(path)
        if 'case' not in saved:
            raise ValueError(f'unknown case-ID schema: {path}')
        ids = set(map(int, saved.case))
        excluded |= ids
        receipts.append(dict(path=str(path), sha256=sha256(path), kind='explicit_case_csv', cases=len(ids)))
    for path in sorted(old_root.glob('results_experiments/**/dataset_split_ids.json')):
        data = json.loads(path.read_text())
        ids = set(int(c) for v in data.get('splits', {}).values() for c in v)
        excluded |= ids
        receipts.append(dict(path=str(path), sha256=sha256(path), kind='explicit_ids', cases=len(ids)))
    seen = set()
    for path in sorted(old_root.glob('icml2027/queue/**/manifest*.json')):
        data = json.loads(path.read_text())
        for cell in data.get('cells', []):
            key = (cell['cluster'], cell['seed'], canonical_hash(cell['panel_sizes']))
            if key in seen:
                continue
            seen.add(key)
            ex = set()
            for pattern in cell.get('exclude_globs', []):
                for p in old_root.glob(pattern):
                    ex.update(int(c) for v in json.loads(p.read_text()).get('splits', {}).values() for c in v)
            panels = make_panels(df, cell['cluster'], cell['seed'], cell['panel_sizes'], ex)
            excluded.update(int(c) for name in ('train', 'opt', 'cal', 'eval') for c in getattr(panels, name).case)
        receipts.append(dict(path=str(path), sha256=sha256(path), kind='reconstructed_manifests'))
    # Published baseline tables also consumed labels: headline seeds 4/5 and
    # repeated split-noise panels cannot silently re-enter the new holdout.
    # The local repository also retains result tables from other hosts. Include
    # the bundled metadata snapshot even when Euler has no copy of those tables.
    baseline_source = ROOT if (ROOT / 'icml2027/results/baselines').exists() else old_root
    cfg_path = baseline_source / 'icml2027/configs/panels.yaml'
    if cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
        baseline_root = baseline_source / 'icml2027/results/baselines'
        ex = set()
        for pattern in cfg.get('exclude_case_ids_from', []):
            for p in old_root.glob(pattern):
                ex.update(int(c) for v in json.loads(p.read_text()).get('splits', {}).values() for c in v)
        for c in cfg['clusters']:
            seeds = [s for s in cfg['seeds_headline'] if (baseline_root / f'c{c}__s{s}.json').exists()]
            if (baseline_root / f'splitnoise_c{c}.json').exists():
                seeds += list(range(101, 101 + cfg['baseline_redraws']))
            for s in seeds:
                panels = make_panels(df, c, s, cfg['panel_sizes'], ex)
                excluded.update(int(i) for role in ('train', 'opt', 'cal', 'eval') for i in getattr(panels, role).case)
            receipts.append(dict(path=str(cfg_path), sha256=sha256(cfg_path), kind='historical_baseline_panels', cluster=c, seeds=seeds))
        if list(baseline_root.glob('*.json')):
            excluded.update(map(int, df.sample(n=min(20000, len(df)), random_state=0).case))
            receipts.append(dict(kind='baseline_global_training_sample', seed=0, rows=min(20000, len(df)),
                                 source='src/optim/baselines.py::main'))
    arrays = sorted(set(p for base in ('arr2026/results', 'arr2026/results_euler', 'arr2026/results_hse', 'arr2026/results_v100', 'results_euler', 'results_hse')
                        for p in (old_root / base).glob('**/*answers.npy')))
    skipped, cache, saved_hashes = [], {}, {}
    for path in arrays:
        # Other instruments have separate ID namespaces. Never match them to IPIP.
        summary = path.parent / 'summary.json'
        info = json.loads(summary.read_text()) if summary.exists() else {}
        corpus = str(info.get('corpus', info.get('instrument', ''))).lower()
        if any(k in str(path).lower() for k in ('ipip300', 'ipip_300', 'big5', 'hexaco', 'sd3')) or corpus in ('big5', 'hexaco', 'ipip300', 'sd3'):
            skipped.append(dict(path=str(path), reason='different instrument'))
            continue
        digest = sha256(path)
        if digest in saved_hashes:
            receipts.append(dict(path=str(path), sha256=digest, kind='identical_array', source=saved_hashes[digest]))
            continue
        saved = np.load(path, allow_pickle=False)
        if saved.ndim != 2:
            raise ValueError(f'unknown historical answer tensor: {path}, {saved.shape}')
        ids = info.get('target_ids') if 'target' in path.name or 'train' in path.name else None
        if ids is None:
            ids = list(range(1, 121)) if saved.shape[1] == 120 else ITEM_SPLIT[1] if saved.shape[1] == 60 else None
        if ids is None or len(ids) != saved.shape[1]:
            raise ValueError(f'cannot identify historical exposure items: {path}')
        cases, rec = fingerprint_exposures(df, saved, ids, cache)
        saved_hashes[digest] = str(path)
        excluded |= cases
        receipts.append(dict(path=str(path), sha256=sha256(path), kind='response_fingerprint', item_ids=ids, **rec))
    if not seen or not arrays:
        raise ValueError('incomplete history inventory: manifests and historical arrays required')
    return excluded, dict(sources=receipts, skipped=skipped, historical_panels=len(seen),
                          excluded_cases=len(excluded), coverage='all discoverable saved IPIP120 answer arrays and queued ICML panels; no blanket claim beyond inventoried paths')


def prepare(old_root, output, manifest):
    output = Path(output)
    if (output / 'panels.json').exists():
        raise FileExistsError('immutable panels already exist')
    corpus_path = Path(old_root) / 'data/raw/df_ipipneo_120_clusters'
    df = load_corpus(corpus_path)
    excluded, receipt = collect_exposures(df, old_root)
    spec = json.loads(Path(manifest).read_text())
    splits = global_split(df, spec['panel_sizes'], spec['split_seed'], excluded)
    blob = dict(schema='hbs-global-v2', source_sha256=sha256(corpus_path), splits=splits,
                exposure_receipt_sha256=canonical_hash(receipt), cohort='complete-case IPIP120, no cluster routing')
    blob['sha256'] = canonical_hash(blob)
    atomic_write_json(output / 'exposure_receipt.json', receipt)
    atomic_write_json(output / 'panels.json', blob)
    # Preserve private ID evidence on cluster only.
    atomic_write_json(output / 'excluded_ids.json', sorted(excluded))
    atomic_write_json(output / 'READY.json', dict(manifest_sha256=sha256(manifest), panels_sha256=blob['sha256'],
                                                scientific_result=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--old-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--manifest', required=True)
    a = p.parse_args()
    prepare(a.old_root, a.output, a.manifest)
