import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.optim.genotype import seed_genotype  # noqa: E402

TEMPLATE = seed_genotype(0)


class JitterMutator:
    """Returns a parsable variant of whatever persona JSON appears in the prompt."""

    def __init__(self, seed=0, fail_every=0):
        self.rng = np.random.default_rng(seed)
        self.n = 0
        self.fail_every = fail_every

    def complete(self, system, user, temperature=0.7, max_tokens=1200):
        self.n += 1
        if self.fail_every and self.n % self.fail_every == 0:
            return "sorry, no json", 10, 5
        if "Write a different instruction" in user:
            return "Make each facet sentence more concrete.", 5, 5
        g = copy.deepcopy(TEMPLATE)
        g["role_definition"] += " " + "x" * int(self.rng.integers(0, 60))
        return json.dumps({k: g[k] for k in ("role_definition", "trait_formulations", "facet_formulations",
                                             "critic_formulations")}), 100, 50


class PersonaReadout:
    """Deterministic fake: beliefs depend on the persona text so different prompts score differently."""

    def beliefs(self, systems, item_ids):
        n, J = len(systems), len(item_ids)
        P = np.zeros((n, J, 5))
        for a, s in enumerate(systems):
            h = (len(s) % 97) / 97.0
            rng = np.random.default_rng(len(s) * 31 + a)
            P[a] = rng.dirichlet(np.ones(5) * (1 + 3 * h), size=J)
        return P, np.full((n, J), 0.98), 50 * n * J


def toy_corpus(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.integers(1, 6, size=(n, 120)).astype(float), columns=[f"i{i}" for i in range(1, 121)])
    df["case"] = np.arange(n) + 10_000_000
    df["clusters"] = rng.integers(0, 2, size=n)
    return df


@pytest.fixture
def corpus():
    return toy_corpus()
