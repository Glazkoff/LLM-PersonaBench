import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEYS = ("role_definition", "trait_formulations", "facet_formulations", "critic_formulations")


class GenotypeParseError(ValueError):
    pass


def to_json(g: dict) -> str:
    return json.dumps({k: g[k] for k in KEYS if k in g}, sort_keys=True, ensure_ascii=False, indent=1)


def _first_object(text: str) -> dict:
    s = text.find("{")
    while s != -1:
        depth = 0
        for i in range(s, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[s:i + 1])
                    except json.JSONDecodeError:
                        break
        s = text.find("{", s + 1)
    raise GenotypeParseError("no JSON object found")


def parse_genotype(text: str, template: dict) -> dict:
    obj = _first_object(text or "")
    if not isinstance(obj, dict) or any(k not in obj for k in KEYS):
        raise GenotypeParseError("missing required keys")
    out = copy.deepcopy(template)
    missing = total = 0
    for k in ("trait_formulations", "facet_formulations"):
        src = obj.get(k) if isinstance(obj.get(k), dict) else {}
        for name in template[k]:
            total += 1
            v = src.get(name)
            if isinstance(v, str) and v.strip():
                out[k][name] = v.strip()
            else:
                missing += 1
    if total and missing / total > 0.5:
        raise GenotypeParseError("more than half the entries missing")
    for k in ("role_definition", "critic_formulations"):
        if not isinstance(obj[k], str) or not obj[k].strip():
            raise GenotypeParseError(f"{k} empty")
        out[k] = obj[k].strip()
    return out


def seed_genotype(cluster: int) -> dict:
    sysm = json.loads((ROOT / "src/prompt/system.json").read_text())
    tr = json.loads((ROOT / "src/prompt/previous_base_prompt/traits.json").read_text())[str(cluster)]
    fc = json.loads((ROOT / "src/prompt/previous_base_prompt/facets.json").read_text())[str(cluster)]
    return {"role_definition": sysm["role"], "trait_formulations": dict(tr), "facet_formulations": dict(fc),
            "critic_formulations": sysm["critic_internal"], "intensity_modifiers": sysm["intensity_modifiers"]}
