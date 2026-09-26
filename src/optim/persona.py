import numpy as np

from src.optim.panels import FACET_NAME
from src.utils.prompt import get_modifier_bisect

TRAIT_OF = {"N": "neuroticism", "E": "extraversion", "O": "openness", "A": "agreeableness",
            "C": "conscientiousness"}


def _norm(s):
    return s.lower().replace("facet_", "").replace("-", "_").replace(" ", "_")


def system_prompt(genotype: dict, scores: dict) -> str:
    mods = genotype["intensity_modifiers"]
    trait_val = {TRAIT_OF[t]: float(np.mean([v for fk, v in scores.items() if fk[0] == t])) for t in TRAIT_OF}
    by_name = {_norm(FACET_NAME[fk]): v for fk, v in scores.items()}
    lines = [f"- This trait ({t}) describes you {get_modifier_bisect(trait_val[t], mods)}: {d}"
             for t, d in genotype["trait_formulations"].items() if t in trait_val]
    for f, d in genotype["facet_formulations"].items():
        v = by_name.get(_norm(f))
        if v is not None:
            lines.append(f"- This facet ({f}) describes you {get_modifier_bisect(v, mods)}: {d}")
    return (f"{genotype['role_definition']}\nYour profile:\n" + "\n".join(lines)
            + f"\n\nInternal reflection guideline:\n{genotype['critic_formulations']}")
