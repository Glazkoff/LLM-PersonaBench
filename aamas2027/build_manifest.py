"""Expand an immutable full-comparison matrix; never infer cells from results."""
import argparse
import json
from pathlib import Path

from aamas2027.pipeline import canonical_hash
from src.optim.results import atomic_write_json

ARMS = ('ga', 'de', 'promptbreeder', 'mapelites', 'gepa', 'gepa_official', 'gigaevo_native',
        'opro', 'protegi', 'random', 'bestofb')
MODELS = ('Qwen/Qwen3.6-35B-A3B', 'ibm-granite/granite-4.2-30b')


def build():
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root / 'icml2027/queue/euler/manifest_tier1.json').read_text())
    cfg = {c['arm']: c['arm_cfg'] for c in old['cells']}
    cfg['gepa_official'] = dict(reflection_minibatch_size=8, max_tokens=2500)
    cfg['gigaevo_native'] = dict(max_mutants=160, max_tokens=2500)
    # All methods use one common max output cap large enough for 35 JSON fields.
    for c in cfg.values():
        c.update(max_tokens=2500, temperature=0.7)
    cells = []
    for model in MODELS:
        for seed in (1, 2, 3):
            variants = [(arm, 'rps_raw') for arm in ARMS] + [('ga', 'rps_cal'), ('ga', 's0'), ('mapelites', 'rps_cal')]
            for arm, objective in variants:
                label = arm if objective == 'rps_raw' else arm + '_' + objective
                cells.append(dict(cell_id=f'{label}__{model.replace("/", "_")}__s{seed}', arm=arm,
                                  arm_label=label, hf_id=model, seed=seed, fitness=objective,
                                  arm_cfg=cfg[arm], budget_B=80))
    spec = dict(schema='hbs-aamas-global-v2', cohort='IPIP120-global-complete-case', split_seed=20261004,
                panel_sizes=dict(train=8000, opt=80, select=120, cal=120, eval=600),
                scorer='v2', alphas=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.2, 1.4, 1.6, 1.8, 2],
                primary_objective='rps_raw', primary_final_metric='rps_cal',
                primary_method='mapelites', comparator_selection='lowest mean selection RPS across seeds per model, excluding primary and objective ablations',
                selection_candidates=5, mutator='Qwen/Qwen3.8-27B', prefill='My answer is ',
                seeds=[1, 2, 3], cells=cells,
                actual_budget='80 * N_opt * N_target executed forecasts; invalid proposals do not spend forecast budget',
                native_sources=dict(gepa='8b0ce6cd99a234f6b74daf37558a2ac0ce18f975',
                                    gigaevo_internal='f72ab08c628cc49f4c4c1b6ee48bd11b68d3ce08'),
                labels=dict(gepa='GEPA-style local', mapelites='MAP-Elites-style local',
                            gepa_official='official GEPA v0.1.4', gigaevo_native='native GigaEvo internal v2.20.1'))
    spec['sha256'] = canonical_hash(spec)
    return spec


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError('manifest is immutable')
    atomic_write_json(a.output, build())
