from src.optim.arms import ops
from src.optim.arms.base import Arm


class RandomSearch(Arm):
    """B-1 independent mutations of the seed; best on the optimisation panel."""
    name = "random"

    def search(self, seed):
        sid, _ = self._eval(seed, 0, [], "init")
        while self._remaining() > 0:
            self._mutate(seed, 1, [sid])
        return self.best_genotype


class ParaphraseSearch(Arm):
    """Chained paraphrases with NO selection pressure; returns the last parsed one."""
    name = "paraphrase"

    def search(self, seed):
        cur = seed
        cur_id, _ = self._eval(cur, 0, [], "init")
        while self._remaining() > 0:
            nxt = self._ask(ops.SYSTEM, ops.PARAPHRASE.format(genotype=self._gjson(cur)), fallback=cur)
            nid, _ = self._eval_asked(nxt, 1, [cur_id], "paraphrase")
            if self.last_parse_ok:
                cur, cur_id = nxt, nid
        return cur


class BestOfB(Arm):
    """Selection without evolution: samples from the initial-population generator at T=1.0."""
    name = "bestofb"

    def search(self, seed):
        sid, _ = self._eval(seed, 0, [], "init")
        while self._remaining() > 0:
            self._mutate(seed, 0, [sid], op="init_sample", temperature=1.0)
        return self.best_genotype
