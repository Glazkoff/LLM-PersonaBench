import numpy as np

from src.optim.arms import ops
from src.optim.arms.base import Arm


class OPRO(Arm):
    """LLM-as-optimizer over the scored trajectory (Yang et al.)."""
    name = "opro"

    def _history_block(self, k):
        rows = sorted((h for h in self.history if h["parse_ok"]), key=lambda h: h["rps_cal_opt"])[:k]
        return "\n\n".join(f"score {h['rps_cal_opt']:.4f}\n{self._gjson(h['genotype'])}" for h in reversed(rows))

    def search(self, seed):
        self._eval(seed, 0, [], "init")
        rnd = 1
        while self._remaining() > 0:
            for _ in range(int(self.cfg.get("proposals_per_round", 8))):
                if self._remaining() == 0:
                    break
                g = self._ask(ops.SYSTEM, ops.OPRO.format(history=self._history_block(int(self.cfg.get("history_shown", 12)))),
                              fallback=self.best_genotype)
                self._eval_asked(g, rnd, [self.best_candidate_id], "opro_proposal")
            rnd += 1
        return self.best_genotype


class ProTeGi(Arm):
    """Textual-gradient beam search (Pryzant et al.). Feedback comes from the charged evaluation's cached beliefs."""
    name = "protegi"

    def search(self, seed):
        sid, s = self._eval(seed, 0, [], "init")
        beam, B, rnd = [(sid, s, seed)], int(self.cfg.get("beam", 4)), 1
        while self._remaining() > 0:
            cands = list(beam)
            for cid, _, g in beam:
                for _ in range(int(self.cfg.get("gradients_per_candidate", 2))):
                    if self._remaining() == 0:
                        break
                    fb = self.fitness.feedback(g, k=10)
                    child = self._ask(ops.SYSTEM, ops.REFLECT.format(genotype=self._gjson(g), feedback=fb), fallback=g)
                    ncid, nsc = self._eval_asked(child, rnd, [cid], "textual_gradient_edit")
                    if self.last_parse_ok:
                        cands.append((ncid, nsc, child))
            cands.sort(key=lambda t: t[1])
            beam = cands[:B]
            rnd += 1
        return self.best_genotype


def pareto_front(scores):
    ids = list(scores)
    M = np.stack([scores[i] for i in ids])
    winners = np.argmin(M, axis=0)
    counts = {ids[c]: int((winners == c).sum()) for c in range(len(ids))}
    front = [i for i in ids if counts[i] > 0]
    return front, {i: counts[i] for i in front}


class GEPA(Arm):
    """Reflective prompt evolution with a per-respondent Pareto pool (Agrawal et al., ICLR 2026)."""
    name = "gepa"

    def search(self, seed):
        n_opt = self.fitness.n_opt
        mb = int(self.cfg.get("minibatch_respondents", 8))
        frac = mb / n_opt
        passes = mb * (self.fitness.forward_passes_per_candidate // n_opt)
        sid, _ = self._eval(seed, 0, [], "init")
        pool = {sid: (seed, self.fitness.per_respondent_scores(seed))}
        rnd = 1
        while self.ledger.remaining_fraction() >= 2 * frac + 1:
            front, wts = pareto_front({k: v[1] for k, v in pool.items()})
            p = np.array([wts[k] for k in front], float)
            pid = front[self.rng.choice(len(front), p=p / p.sum())]
            parent, parent_per = pool[pid]
            idx = self.rng.choice(n_opt, size=mb, replace=False)
            self.fitness.evaluate_subset(parent, idx)
            self.ledger.charge_fraction(frac, passes, prompt_tokens=self.fitness.last_prompt_tokens)
            fb = self.fitness.feedback(parent, k=int(self.cfg.get("reflection_items_shown", 10)), prefer_subset=True)
            child = self._ask(ops.SYSTEM, ops.REFLECT.format(genotype=self._gjson(parent), feedback=fb), fallback=parent)
            if not self.last_parse_ok:
                self._eval_asked(child, rnd, [pid], "gepa_reflect_unparsable")
                rnd += 1
                continue
            sub = self.fitness.evaluate_subset(child, idx)
            self.ledger.charge_fraction(frac, passes, prompt_tokens=self.fitness.last_prompt_tokens)
            if sub.rps_cal_opt < float(np.nanmean(parent_per[idx])):
                cid, _ = self._eval_asked(child, rnd, [pid], "gepa_reflect_full")
                pool[cid] = (child, self.fitness.per_respondent_scores(child))
                if len(pool) > int(self.cfg.get("pareto_pool_max", 12)):
                    keep = set(pareto_front({k: v[1] for k, v in pool.items()})[0]) | {self.best_candidate_id}
                    pool = {k: v for k, v in pool.items() if k in keep}
            rnd += 1
        # spend the remainder on full evaluations of mutations of the best, so every arm exhausts B
        while self._remaining() > 0 and self.ledger.remaining_fraction() >= 1:
            self._mutate(self.best_genotype, rnd, [self.best_candidate_id], op="gepa_fill_mutation")
        rest = self.ledger.remaining_fraction()
        if rest > 1e-9:
            self.ledger.charge_fraction(rest)
        return self.best_genotype
