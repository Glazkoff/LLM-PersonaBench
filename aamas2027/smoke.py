"""Real local-model protocol smoke on eight opt people; no evaluation access."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from aamas2027.build_manifest import ARMS as NAMES
from aamas2027.native_gigaevo import NativeGigaEvo
from aamas2027.pipeline import neutral_genotype
from aamas2027.runner import MeteredReadout, load_context
from src.optim.arms import ARMS
from src.optim.arms.base import LLMMutator
from src.optim.budget import BudgetLedger
from src.optim.fitness import Fitness
from src.optim.genotype import to_json
from src.optim.results import CellWriter, atomic_write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--results', required=True)
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--old-root', required=True)
    a = p.parse_args()
    man, cell, panels, parts, root, blob = load_context(a.manifest, a.results, a.old_root, a.index, 'search')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ['HBS_MODEL_SNAPSHOT'], trust_remote_code=True, local_files_only=True)
    readout = MeteredReadout(cell['hf_id'], tokenizer=tok, workers=2, batch=16)
    panels.opt = panels.opt.iloc[:8]
    seed = neutral_genotype()
    output = root / 'smoke' / cell['hf_id'].replace('/', '_') / os.environ['SLURM_JOB_ID']
    summaries = {}
    for name in NAMES:
        fit = Fitness(panels, readout, man['alphas'], np.random.default_rng(man['split_seed']), objective='rps_raw')
        cfg = next(c['arm_cfg'] for c in man['cells'] if c['arm'] == name)
        cfg = {**cfg, 'seed': 999, 'strict_genotype': True, 'minibatch_respondents': 2,
               'population_size': 2, 'max_mutants': 12, 'max_invalid_proposals': 8, 'timeout_s': 2400}
        mutator = LLMMutator(man['mutator'])
        arm = (NativeGigaEvo if name == 'gigaevo_native' else ARMS[name])(cfg, np.random.default_rng(999), mutator, seed)
        ledger = BudgetLedger(4)
        writer = CellWriter(output, name)
        before = readout.confirmed_forecasts
        result = arm.run(fit, seed, ledger, writer)
        actual = readout.confirmed_forecasts - before
        valid = [h for h in arm.history if h.get('parse_ok') and h.get('full_opt_panel', True)]
        if actual != ledger.forward_passes or not (0 < actual <= 4 * fit.forward_passes_per_candidate):
            raise RuntimeError(f'smoke budget mismatch: {name}')
        if len({to_json(h['genotype']) for h in valid}) < 2:
            raise RuntimeError(f'{name}: no newly generated valid full-panel candidate in smoke')
        masses = [h['mass_on_scale'] for h in valid if 'mass_on_scale' in h]
        if masses and min(masses) < .5:
            raise RuntimeError(f'{name}: invalid probability-readout mass')
        summaries[name] = dict(budget=ledger.to_dict(), selected_sha=to_json(result), valid_candidates=len(valid))
    if readout.failed_requests or readout.requested_forecasts != readout.confirmed_forecasts:
        raise RuntimeError('unaccounted forecast request')
    atomic_write_json(root / 'smoke' / (cell['hf_id'].replace('/', '_') + '.PASSED.json'),
                      dict(scientific_evidence=False, arms=summaries, cost=readout.receipt(),
                           panel_sha256=blob['sha256'], manifest_sha256=man['sha256']))


if __name__ == '__main__':
    main()
