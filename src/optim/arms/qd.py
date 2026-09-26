import bisect

import numpy as np

from src.optim.arms import ops
from src.optim.arms.base import Arm
from src.optim.arms.evolutionary import GA


def bin_index(v, edges):
    if v != v:  # NaN
        return 0
    return max(0, min(bisect.bisect_right(edges, v) - 1, len(edges) - 2))


class Archive:
    def __init__(self):
        self.cells = {}

    def add(self, key, cid, score, genotype, descriptor):
        cur = self.cells.get(key)
        if cur is None or score < cur[1]:
            self.cells[key] = (cid, score, genotype, descriptor)
            return True
        return False

    def sample(self, rng):
        keys = list(self.cells)
        return self.cells[keys[rng.integers(len(keys))]]


class MAPElites(Arm):
    """GigaEvo-style quality-diversity: per-island MAP-Elites over (prompt length, predicted dispersion),
    lineage-insight-conditioned mutation, periodic migration of island champions."""
    name = "mapelites"

    def _key(self, d):
        b = self.cfg["archive_bins"]
        return tuple(bin_index(d[k], b[k]) for k in sorted(b))

    def _insights(self, lineage):
        w = int(self.cfg.get("insight_window", 6))
        return "\n".join(f"- {r['operation']}: score {r['rps_cal_opt']:.4f}, length {r.get('descriptor', {}).get('prompt_length_chars')}, "
                         f"predicted sd {r.get('descriptor', {}).get('predicted_sd', float('nan')):.2f}" for r in lineage[-w:])

    def _record(self, g, gen, parents, op, isl, ok, toks):
        cid, sc = self._eval(g, gen, parents, op, parse_ok=ok, mutator_tokens=toks)
        d = self.fitness.last_descriptor() if ok else {"prompt_length_chars": len(self._gjson(g)), "predicted_sd": float("nan")}
        self.history[-1]["descriptor"] = d
        self.history[-1]["island"] = isl
        return cid, sc, d

    def search(self, seed):
        n = int(self.cfg.get("islands", 2))
        arch = [Archive() for _ in range(n)]
        lin = [[] for _ in range(n)]
        for isl in range(n):
            for i in range(int(self.cfg.get("init_per_island", 4))):
                if self._remaining() == 0:
                    break
                if isl == 0 and i == 0:
                    g, ok, toks, op = seed, True, (0, 0), "init"
                else:
                    g = self._ask(ops.SYSTEM, ops.MUTATE.format(genotype=self._gjson(seed)), fallback=seed)
                    ok, toks, op = self.last_parse_ok, self.last_mutator_tokens, "init_mutation"
                cid, sc, d = self._record(g, 0, [], op, isl, ok, toks)
                if ok:
                    arch[isl].add(self._key(d), cid, sc, g, d)
                lin[isl].append(self.history[-1])
        it = 1
        while self._remaining() > 0:
            isl = it % n
            if not arch[isl].cells:
                arch[isl].add((0, 0), self.best_candidate_id, self.best_score, self.best_genotype, {})
            pid, _, pg, _ = arch[isl].sample(self.rng)
            user = (ops.MUTATE.format(genotype=self._gjson(pg))
                    + "\n\nLINEAGE INSIGHTS (recent attempts on this island; lower score is better):\n" + self._insights(lin[isl]))
            g = self._ask(ops.SYSTEM, user, fallback=pg)
            cid, sc, d = self._record(g, it, [pid], "qd_mutation", isl, self.last_parse_ok, self.last_mutator_tokens)
            if self.last_parse_ok:
                arch[isl].add(self._key(d), cid, sc, g, d)
            lin[isl].append(self.history[-1])
            if n > 1 and it % int(self.cfg.get("migrate_every", 12)) == 0:
                best = min(arch[isl].cells.values(), key=lambda t: t[1])
                arch[(isl + 1) % n].add(self._key(best[3]) if best[3] else (0, 0), *best)
            it += 1
        return self.best_genotype


def nondominated_sort(F):
    n = len(F)
    S = [[] for _ in range(n)]
    cnt = np.zeros(n, int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                S[p].append(q)
            elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                cnt[p] += 1
        if cnt[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                cnt[q] -= 1
                if cnt[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return [f for f in fronts if f]


def crowding_distance(F, front):
    if len(front) <= 2:
        return {i: np.inf for i in front}
    d = {i: 0.0 for i in front}
    for m in range(F.shape[1]):
        order = sorted(front, key=lambda i: F[i, m])
        rng = (F[order[-1], m] - F[order[0], m]) or 1.0
        d[order[0]] = d[order[-1]] = np.inf
        for k in range(1, len(order) - 1):
            d[order[k]] += (F[order[k + 1], m] - F[order[k - 1], m]) / rng
    return d


def knee_point(F, front):
    P = F[front]
    a, b = P[np.argmin(P[:, 0])], P[np.argmin(P[:, 1])]
    if np.allclose(a, b):
        return front[int(np.argmin(P[:, 0]))]
    ab = (b - a) / np.linalg.norm(b - a)
    dist = [np.linalg.norm((p - a) - np.dot(p - a, ab) * ab) for p in P]
    return front[int(np.argmax(dist))]


class NSGA2(GA):
    """Multi-objective (calibrated RPS, -individuation) with non-dominated sorting and crowding (Deb et al. 2002).
    Returns the knee of the final front; the whole front goes to pareto_front.json."""
    name = "nsga2"

    def _obj(self):
        o = self.fitness.last_objectives()
        return np.array([o["rps_cal"], o["neg_vr_between"]], float)

    def search(self, seed):
        P = int(self.cfg["population_size"])
        pop = []  # (cid, score, genotype, objectives)
        sid, s = self._eval(seed, 0, [], "init")
        pop.append((sid, s, seed, self._obj()))
        while len(pop) < P and self._remaining() > 0:
            cid, sc, g, ok = self._mutate(seed, 0, [sid], op="init_mutation")
            if ok:
                pop.append((cid, sc, g, self._obj()))
        gen = 1
        while self._remaining() > 0:
            kids = []
            while len(kids) < P and self._remaining() > 0:
                o = self._offspring([t[:3] for t in pop], gen)
                if o:
                    kids.append((*o, self._obj()))
            allp = pop + kids
            F = np.array([t[3] for t in allp])
            keep = []
            for fr in nondominated_sort(F):
                if len(keep) + len(fr) <= P:
                    keep += fr
                else:
                    cd = crowding_distance(F, fr)
                    keep += sorted(fr, key=lambda i: -cd[i])[: P - len(keep)]
                    break
            pop = [allp[i] for i in keep]
            gen += 1
        F = np.array([t[3] for t in pop])
        front = nondominated_sort(F)[0]
        self.writer.write_front([{"candidate_id": pop[i][0], "objectives": F[i].tolist(),
                                  "genotype": {k: pop[i][2][k] for k in ("role_definition", "trait_formulations",
                                                                         "facet_formulations", "critic_formulations")}}
                                 for i in front])
        return pop[knee_point(F, front)][2]
