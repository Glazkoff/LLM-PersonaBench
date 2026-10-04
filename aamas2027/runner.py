"""Search -> common held-out selection -> global freeze -> one final evaluation.

Search code receives empty cal/eval frames. Test forecasts cannot be generated
until all 84 selected candidates and the strongest comparator are hash-frozen.
"""
import argparse
import copy
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from aamas2027.pipeline import canonical_hash, neutral_genotype, sha256
from src.optim import scoring as S
from src.optim.arms import ARMS
from src.optim.arms.base import LLMMutator
from src.optim.budget import BudgetLedger
from src.optim.fitness import Fitness
from src.optim.frozen_eval import beliefs_for
from src.optim.genotype import parse_genotype, to_json
from src.optim.panels import ITEM_SPLIT, Panels, answers, load_corpus
from src.optim.readout import VLLMReadout
from src.optim.results import CellWriter, atomic_write_json


class MeteredReadout(VLLMReadout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Failed HTTP requests have unknown server-side cost: abort the cell,
        # log the uncertainty, never silently retry and hide extra inference.
        self.client = self.client.with_options(max_retries=0)
        self.lock = threading.Lock()
        self.requested_forecasts = self.confirmed_forecasts = self.failed_requests = 0
        self.prompt_tokens = 0

    def _batch(self, prompts):
        with self.lock:
            self.requested_forecasts += len(prompts)
        try:
            values, tokens = super()._batch(prompts)
            if len(values) != len(prompts) or any(v is None for v in values):
                raise ValueError('incomplete readout response')
            with self.lock:
                self.confirmed_forecasts += len(prompts)
                self.prompt_tokens += tokens
            return values, tokens
        except Exception:
            with self.lock:
                self.failed_requests += 1
            raise

    def receipt(self):
        return dict(requested_forecasts=self.requested_forecasts, confirmed_forecasts=self.confirmed_forecasts,
                    failed_requests=self.failed_requests, prompt_tokens=self.prompt_tokens,
                    uncertain_forecasts=self.requested_forecasts - self.confirmed_forecasts)


def load_context(manifest, results, old_root, index, phase):
    man = json.loads(Path(manifest).read_text())
    assert canonical_hash({k: v for k, v in man.items() if k != 'sha256'}) == man['sha256']
    root = Path(results)
    blob = json.loads((root / 'panels.json').read_text())
    assert canonical_hash({k: v for k, v in blob.items() if k != 'sha256'}) == blob['sha256']
    ready = json.loads((root / 'READY.json').read_text())
    if ready['manifest_sha256'] != sha256(manifest) or ready['panels_sha256'] != blob['sha256']:
        raise ValueError('preflight provenance mismatch')
    corpus = Path(old_root) / 'data/raw/df_ipipneo_120_clusters'
    if sha256(corpus) != blob['source_sha256']:
        raise ValueError('corpus changed after preflight')
    df = load_corpus(corpus).set_index('case', drop=False)
    empty = df.iloc[:0]
    allowed = ('train', 'opt', 'select') if phase == 'search' else ('train', 'cal', 'eval')
    parts = {k: df.loc[blob['splits'][k]] if k in allowed else empty for k in blob['splits']}
    cell = man['cells'][index]
    panels = Panels(-1, cell['seed'], train=parts['train'], opt=parts['opt'], cal=parts['cal'], eval=parts['eval'])
    return man, cell, panels, parts, root, blob


def candidates(history, returned, limit):
    by = {}
    for row in history:
        if (row.get('parse_ok', True) and row.get('full_opt_panel', True)
                and np.isfinite(row.get('rps_cal_opt', np.inf)) and row.get('genotype') is not None):
            by[to_json(row['genotype'])] = (row['rps_cal_opt'], row['genotype'])
    if to_json(returned) not in by:
        raise ValueError('returned genotype has no recorded full-panel evaluation')
    out = [returned]
    seen = {to_json(returned)}
    for _, g in sorted(by.values(), key=lambda x: (x[0], to_json(x[1]))):
        if to_json(g) not in seen:
            out.append(parse_genotype(to_json(g), neutral_genotype()))
            seen.add(to_json(g))
        if len(out) >= limit:
            break
    return out


def search(man, cell, panels, parts, root, blob, readout):
    w = CellWriter(root / 'cells', cell['cell_id'])
    if (w.dir / 'selected.json').exists():
        return
    if (w.dir / 'history.jsonl').exists():
        raise FileExistsError('partial search exists; preserve it and explicitly create a new attempt')
    w.write_config(cell)
    w.set_status('search_running', slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    seed = neutral_genotype()
    # Fold seed is fixed across optimization random seeds as well as methods.
    fit = Fitness(panels, readout, man['alphas'], np.random.default_rng(man['split_seed']), objective=cell['fitness'])
    mutator = LLMMutator(man['mutator'], base_url=os.environ.get('MUTATOR_BASE_URL'))
    rng = np.random.default_rng(cell['seed'])
    ledger = BudgetLedger(cell['budget_B'])
    cfg = {**cell['arm_cfg'], 'seed': cell['seed'], 'strict_genotype': True, 'fatal_mutator_errors': True}
    if cell['arm'] == 'gigaevo_native':
        from aamas2027.native_gigaevo import NativeGigaEvo
        arm = NativeGigaEvo(cfg, rng, mutator, seed)
    else:
        arm = ARMS[cell['arm']](cfg, rng, mutator, seed)
    before = readout.receipt()
    t0 = time.monotonic()
    returned = arm.run(fit, seed, ledger, w)
    if mutator.failed_calls:
        raise RuntimeError('mutator requests failed with unknown token cost; this search is not comparable')
    cost = readout.receipt()
    actual = cost['confirmed_forecasts'] - before['confirmed_forecasts']
    cap = cell['budget_B'] * fit.forward_passes_per_candidate
    if actual > cap or actual != ledger.forward_passes or cost['failed_requests'] or cost['uncertain_forecasts']:
        raise RuntimeError(f'budget reconciliation failed: actual={actual}, ledger={ledger.forward_passes}, cap={cap}')
    atomic_write_json(w.dir / 'returned.json', returned)
    # All arms get the same separate shortlist-selection cap. It is logged and
    # never presented as part of the B=80 optimization budget.
    ps = Panels(-1, cell['seed'], panels.train, parts['select'], parts['select'].iloc[:0], parts['select'].iloc[:0])
    select_fit = Fitness(ps, readout, man['alphas'], np.random.default_rng(man['split_seed']), objective='rps_raw')
    shortlist = candidates(arm.history, returned, man['selection_candidates'])
    scored = []
    for g in shortlist:
        r = select_fit.evaluate(g)
        scored.append((r.rps_raw_opt, canonical_hash(g), g, r.mass))
    score, _, best, mass = min(scored, key=lambda x: (x[0], x[1]))
    out = dict(genotype=best, selection_score=score, selection_mass=mass,
               candidates_considered=len(shortlist), panel_sha256=blob['sha256'], manifest_sha256=man['sha256'],
               search_cost=cost, total_cost=readout.receipt(), ledger=ledger.to_dict(),
               mutator_requests=(dict(total_native_llm_calls=ledger.mutator_calls, details='native_mutator_cost.json')
                                 if cell['arm'] == 'gigaevo_native' else
                                 dict(requested=mutator.requested_calls, completed=mutator.completed_calls, failed=mutator.failed_calls)),
               search_seconds=time.monotonic() - t0,
               label=man.get('labels', {}).get(cell['arm'], cell['arm']))
    atomic_write_json(w.dir / 'selected.json', out)
    w.set_status('selected_waiting_for_global_freeze')


def final(man, cell, panels, root, blob, readout):
    w = CellWriter(root / 'cells', cell['cell_id'])
    if (w.dir / 'final.json').exists():
        return
    frozen = json.loads((root / 'FROZEN.json').read_text())
    chosen = json.loads((w.dir / 'selected.json').read_text())
    if (frozen['manifest_sha256'] != man['sha256'] or
            frozen['cells'][cell['cell_id']]['sha256'] != canonical_hash(chosen)):
        raise ValueError('selected candidate changed after global freeze')
    tgt = ITEM_SPLIT[1]
    Yc, Ye = answers(panels.cal, tgt), answers(panels.eval, tgt)
    F0 = S.pop_cdf(answers(panels.train, tgt))
    predictions, per, metrics = {}, {}, {}
    for label, g in (('base', neutral_genotype()), ('evolved', chosen['genotype'])):
        Pc, Mc = beliefs_for(readout, g, panels.cal, tgt)
        alpha, _ = S.select_alpha(S.cdf(Pc), F0, Yc, man['alphas'])
        qbar = S.cdf(Pc).mean(0)
        Pe, Me = beliefs_for(readout, g, panels.eval, tgt)
        cells = S.cal_cells(S.cdf(Pe), F0, Ye, alpha, qbar)
        per[label] = S.per_respondent(cells).tolist()
        metrics[label] = dict(rps_cal=float(np.mean(per[label])), rps_raw=float(S.rps(Pe, Ye).mean()),
                              alpha=alpha, mass=float(Me.mean()), low_mass_fraction=float((Me < .5).mean()))
        predictions.update({label + '_cal': Pc, label + '_eval': Pe, label + '_cal_mass': Mc, label + '_eval_mass': Me})
        if label == 'evolved':
            # A derangement avoids unchanged identities; common seed across arms.
            rng = np.random.default_rng(man['split_seed'])
            order = rng.permutation(len(panels.eval))
            perm = np.empty_like(order)
            perm[order] = np.roll(order, 1)
            Pw, Mw = beliefs_for(readout, g, panels.eval.iloc[perm], tgt)
            per['wrong_persona'] = S.per_respondent(S.cal_cells(S.cdf(Pw), F0, Ye, alpha, qbar)).tolist()
            predictions.update(wrong=Pw, wrong_mass=Mw, wrong_permutation=perm)
    prior = np.broadcast_to(S.cdf_to_simplex(F0), (len(Ye), len(tgt), 5))
    per['prior'] = S.per_respondent(S.rps(prior, Ye)).tolist()
    np.savez_compressed(w.dir / 'probabilities.npz', **predictions)
    atomic_write_json(w.dir / 'final.json', dict(metrics=metrics, per_respondent=per,
        cost=readout.receipt(), panel_sha256=blob['sha256'], manifest_sha256=man['sha256'],
        selected_sha256=canonical_hash(chosen), observed_target_cells=int(np.isfinite(Ye).sum()),
        n_respondents=len(Ye), n_target_items=len(tgt), missing_predictions_policy='whole cell fails; no arm-specific respondent deletion'))
    w.set_status('completed')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['search', 'final'])
    p.add_argument('--manifest', required=True)
    p.add_argument('--results', required=True)
    p.add_argument('--old-root', required=True)
    p.add_argument('--index', type=int, required=True)
    a = p.parse_args()
    man, cell, panels, parts, root, blob = load_context(a.manifest, a.results, a.old_root, a.index, a.phase)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.environ['HBS_MODEL_SNAPSHOT'], trust_remote_code=True, local_files_only=True)
    readout = MeteredReadout(cell['hf_id'], prefill=man['prefill'], batch=32, workers=4, tokenizer=tokenizer)
    try:
        if a.phase == 'search':
            search(man, cell, panels, parts, root, blob, readout)
        else:
            final(man, cell, panels, root, blob, readout)
    except Exception as exc:
        w = CellWriter(root / 'cells', cell['cell_id'])
        atomic_write_json(w.dir / (a.phase + '_failure.json'), dict(error=repr(exc), cost=readout.receipt()))
        w.set_status('failed_' + a.phase, error=repr(exc))
        raise


if __name__ == '__main__':
    main()
