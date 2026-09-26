from src.optim.arms import ops
from src.optim.arms.base import Arm

SEED_MUTATION_PROMPTS = [
    "Rewrite the persona so each trait sentence names one observable habit.",
    "Sharpen the contrast between high and low levels of every facet.",
    "Make the role_definition a vivid first-person self-description.",
    "Replace abstract adjectives with concrete everyday behaviours.",
    "Tighten every sentence; remove hedging words.",
    "Add to critic_formulations one rule about answering as this specific person, not an average person.",
    "Reorder facets so the most distinctive ones come first.",
    "Rephrase each facet as what the person tends to do when stressed and when relaxed.",
]


class GA(Arm):
    """EvoPrompt-style GA on the JSON genotype: elitism, tournament selection, LLM crossover then mutation."""
    name = "ga"

    def _init_population(self, seed):
        sid, s = self._eval(seed, 0, [], "init")
        pop = [(sid, s, seed)]
        while len(pop) < int(self.cfg["population_size"]) and self._remaining() > 0:
            cid, sc, g, ok = self._mutate(seed, 0, [sid], op="init_mutation")
            if ok:
                pop.append((cid, sc, g))
        return pop

    def _tournament(self, pop):
        k = min(int(self.cfg.get("tournament_k", 3)), len(pop))
        idx = self.rng.choice(len(pop), size=k, replace=False)
        return min((pop[i] for i in idx), key=lambda t: t[1])

    def _offspring(self, pop, gen):
        a, b = self._tournament(pop), self._tournament(pop)
        if self.rng.random() < float(self.cfg.get("crossover_rate", 0.8)) and a[0] != b[0]:
            child = self._ask(ops.SYSTEM, ops.CROSSOVER.format(a=self._gjson(a[2]), b=self._gjson(b[2])), fallback=a[2])
            op, ok, toks = "crossover", self.last_parse_ok, self.last_mutator_tokens
        else:
            child, op, ok, toks = a[2], "copy", True, (0, 0)
        if (op == "copy" or self.rng.random() < float(self.cfg.get("mutation_rate", 0.3))) and ok:
            child = self._ask(ops.SYSTEM, ops.MUTATE.format(genotype=self._gjson(child)), fallback=child)
            op += "+mutation"
            toks = (toks[0] + self.last_mutator_tokens[0], toks[1] + self.last_mutator_tokens[1])
            ok = self.last_parse_ok
        cid, sc = self._eval(child, gen, [a[0], b[0]], op, parse_ok=ok, mutator_tokens=toks)
        return (cid, sc, child) if ok else None

    def search(self, seed):
        pop = self._init_population(seed)
        P, E, gen = int(self.cfg["population_size"]), int(self.cfg.get("elites", 2)), 1
        while self._remaining() > 0:
            pop.sort(key=lambda t: t[1])
            new = pop[:E]
            while len(new) < P and self._remaining() > 0:
                o = self._offspring(pop, gen)
                if o:
                    new.append(o)
            pop = new if len(new) >= 2 else pop
            gen += 1
        return self.best_genotype


class DE(GA):
    """EvoPrompt DE: for each member x pick distinct a,b,c; LLM applies (a-b) to c, crosses with x; greedy replace."""
    name = "de"

    def search(self, seed):
        pop = self._init_population(seed)
        gen = 1
        while self._remaining() > 0:
            for i in range(len(pop)):
                if self._remaining() == 0:
                    break
                others = [j for j in range(len(pop)) if j != i]
                if len(others) < 3:
                    self._mutate(pop[i][2], gen, [pop[i][0]])
                    continue
                a, b, c = (pop[j] for j in self.rng.choice(others, size=3, replace=False))
                x = pop[i]
                trial = self._ask(ops.SYSTEM, ops.DE_STEP.format(a=self._gjson(a[2]), b=self._gjson(b[2]),
                                                                 c=self._gjson(c[2]), x=self._gjson(x[2])), fallback=x[2])
                cid, sc = self._eval_asked(trial, gen, [x[0], a[0], b[0], c[0]], "de_step")
                if self.last_parse_ok and sc < x[1]:
                    pop[i] = (cid, sc, trial)
            gen += 1
        return self.best_genotype


class PromptBreeder(GA):
    """Fernando et al.: each member carries its own mutation prompt, which is itself mutated by the LLM."""
    name = "promptbreeder"

    def search(self, seed):
        pool = list(SEED_MUTATION_PROMPTS[: int(self.cfg.get("mutation_prompt_pool", 8))])
        sid, s = self._eval(seed, 0, [], "init")
        pop = [(sid, s, seed, pool[0])]
        P = int(self.cfg["population_size"])
        while len(pop) < P and self._remaining() > 0:
            mp = pool[len(pop) % len(pool)]
            g = self._ask(ops.SYSTEM, f"{mp}\n\nPERSONA:\n{self._gjson(seed)}", fallback=seed)
            cid, sc = self._eval_asked(g, 0, [sid], "init_mutation")
            if self.last_parse_ok:
                pop.append((cid, sc, g, mp))
        gen = 1
        while self._remaining() > 0:
            pop.sort(key=lambda t: t[1])
            new = pop[:1]
            while len(new) < P and self._remaining() > 0:
                parent = self._tournament(pop)
                mp = parent[3]
                if self.rng.random() < float(self.cfg.get("hyper_mutation_rate", 0.25)):
                    try:
                        text, _, _ = self.mutator.complete(ops.SYSTEM.split("\n")[0], ops.META_MUTATE.format(mutation_prompt=mp),
                                                           temperature=0.9, max_tokens=200)
                    except Exception:  # noqa: BLE001
                        text = ""
                    if text.strip() and "{" not in text:
                        mp = text.strip()[:400]
                        pool.append(mp)
                child = self._ask(ops.SYSTEM, f"{mp}\n\nPERSONA:\n{self._gjson(parent[2])}", fallback=parent[2])
                cid, sc = self._eval_asked(child, gen, [parent[0]], "pb_mutation")
                if self.last_parse_ok:
                    new.append((cid, sc, child, mp))
            pop = new if len(new) >= 2 else pop
            gen += 1
        return self.best_genotype
