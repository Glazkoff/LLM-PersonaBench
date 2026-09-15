r"""H21: can a prompt search actually reach the region $S_0$ rewards?

App. E once claimed the degenerate optimum is "not reachable" from persona
wordings. That claim was withdrawn because the dispersion of the candidates a
search visits was never measured. This measures it.

$S_0$ is maximised by a point mass, so it rewards under-dispersion. The
question is whether prompt space contains candidates that are both fluent
personas AND under-dispersed. We build a candidate set spanning the dispersion
axis deliberately -- the base genotype, the GA-evolved genotype, and variants
carrying explicit style directives in both directions -- and ask which
candidate each rule would select.

If $S_0$ ranks the low-dispersion variants top while a proper rule does not,
the degenerate region IS reachable by prompt search and optimising $S_0$ over a
richer prompt space is unsafe. If $S_0$'s pick coincides with the proper rule's,
the constraint is real and the paper can say so with evidence instead of
withdrawing the claim.

Nothing here is fabricated: every variant is a real prompt, run through the
same readout as every other cell, and scored on held-out respondents.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.prompt import get_modifier_bisect, get_modifier_by_match  # noqa: E402
from _corpora import load as load_corpus  # noqa: E402

DIGITS = ["1", "2", "3", "4", "5"]
ANSWER = ("Answer with a single number from 1 (very inaccurate) to "
          "5 (very accurate). Answer:")
THRESH = [1, 2, 3, 4]

# Style directives appended to the persona. These are the axis under test:
# "decisive" pushes toward a point mass, "varied" away from it. Both are
# ordinary persona instructions a prompt search could plausibly produce.
DIRECTIVES = {
    "base":      "",
    "decisive":  "\nAnswer decisively. Avoid middle-of-the-scale answers; "
                 "commit to the response that best matches this person.",
    "extreme":   "\nThis person holds firm views and answers at the ends of "
                 "the scale rather than hedging.",
    "consistent":"\nBe maximally consistent: similar items should receive the "
                 "same number.",
    "varied":    "\nReal people vary. Let your answers spread across the scale "
                 "as a real respondent's would, including middle options.",
    "calibrated":"\nAnswer as one specific person would, with the natural "
                 "spread of a single human's questionnaire responses.",
}


def load_system():
    return json.loads((ROOT / "src/prompt/system.json").read_text())


def load_genotypes(sysmsg):
    tr = json.loads((ROOT / "src/prompt/previous_base_prompt/traits.json").read_text())
    fc = json.loads((ROOT / "src/prompt/previous_base_prompt/facets.json").read_text())
    base = {int(c): {"role_definition": sysmsg["role"],
                     "trait_formulations": tr[c], "facet_formulations": fc.get(c, {}),
                     "intensity_modifiers": sysmsg["intensity_modifiers"],
                     "critic_formulations": sysmsg["critic_internal"]} for c in tr}
    src = (ROOT / "src/prompt/best_genotype_giga_evoprompt_iter1"
                  "/gigachat3_best_genotype_evoprompt.py").read_text()
    ns = {"system": sysmsg}
    exec(compile(src, "evolved", "exec"), ns)
    evolved = {int(k.rsplit("_", 1)[1]): v for k, v in ns.items()
               if k.startswith("best_genotype_")}
    return base, evolved


def system_prompt(gen, row, directive):
    mods = gen["intensity_modifiers"]
    out = []
    for key, pref in (("trait_formulations", "trait"), ("facet_formulations", "facet")):
        for name, desc in gen.get(key, {}).items():
            v = row.get(name)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            out.append(f"- This {pref} ({name}) describes you "
                       f"{get_modifier_bisect(v, mods)}: {desc}")
    return (f"{gen['role_definition']}\nYour profile:\n" + "\n".join(out) +
            f"\n\nInternal reflection guideline:\n{gen['critic_formulations']}"
            + directive)


def crps(Q, Y):
    ind = np.stack([(Y <= t).astype(float) for t in THRESH], axis=2)
    return np.where(np.isnan(Y), np.nan, np.nansum((Q - ind) ** 2, axis=2) / 4.0)


def s0(P, Y):
    lv = np.arange(1, 6, dtype=float)
    d = np.abs(lv[None, None, :] - Y[:, :, None]) / 4.0
    return np.where(np.isnan(Y), np.nan, np.nansum(P * (1.0 - d), axis=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-cluster", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=260914)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sysmsg = load_system()
    base, evolved = load_genotypes(sysmsg)
    corpus = load_corpus("ipip")
    items = [{"id": i, "text": corpus.text[i]} for i in corpus.ids]
    flip = np.array([i in corpus.rev for i in corpus.ids], dtype=bool)
    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")

    rng = np.random.default_rng(a.seed)
    rows = []
    for c in sorted(set(base) & set(evolved)):
        pool = df.index[df["clusters"] == c].to_numpy()
        rows += [(c, i) for i in rng.choice(pool, min(a.per_cluster, len(pool)),
                                            replace=False)]
    print(f"{len(rows)} respondents; {len(DIRECTIVES)} directives x 2 genotypes",
          flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True).eval()
    dids = [tok.encode(d, add_special_tokens=False)[0] for d in DIGITS]
    tmpl = getattr(tok, "chat_template", "") or ""
    prefill = ("<|channel|>final<|message|>My answer is "
               if "<|channel|>" in tmpl else "My answer is ")

    results = {}
    for gname, gens in (("base", base), ("evolved", evolved)):
        for dname, directive in DIRECTIVES.items():
            prompts, Y = [], np.full((len(rows), len(items)), np.nan)
            for r, (c, idx) in enumerate(rows):
                sysp = system_prompt(gens[c], df.loc[idx], directive)
                for it in items:
                    msgs = [{"role": "system", "content": sysp},
                            {"role": "user", "content": f"{it['text']}\n{ANSWER}"}]
                    try:
                        t = tok.apply_chat_template(msgs, tokenize=False,
                                                    add_generation_prompt=True,
                                                    enable_thinking=False)
                    except TypeError:
                        t = tok.apply_chat_template(msgs, tokenize=False,
                                                    add_generation_prompt=True)
                    prompts.append(t + prefill)
                Y[r] = corpus.Y[idx]

            P = np.full((len(rows), len(items), 5), np.nan)
            with torch.no_grad():
                for b0 in range(0, len(prompts), a.batch_size):
                    ch = prompts[b0:b0 + a.batch_size]
                    enc = tok(ch, return_tensors="pt", padding=True).to(model.device)
                    pr = torch.softmax(model(**enc).logits[:, -1, :].float(), -1)[:, dids]
                    pr = pr.cpu().numpy()
                    tot = pr.sum(1, keepdims=True)
                    for k in range(len(ch)):
                        g = b0 + k
                        P[g // len(items), g % len(items)] = pr[k] / max(tot[k, 0], 1e-9)
            if corpus.recoded and flip.any():
                P[:, flip, :] = P[:, flip, ::-1]

            lv = np.arange(1, 6, dtype=float)
            mean = (P * lv).sum(-1)
            sd = float(np.sqrt(np.clip((P * lv**2).sum(-1) - mean**2, 0, None)).mean())
            key = f"{gname}/{dname}"
            results[key] = dict(s0=float(np.nanmean(s0(P, Y))),
                                crps=float(np.nanmean(crps(np.cumsum(P, 2)[:, :, :4], Y))),
                                sd=sd)
            print(f"  {key:22s} S0={results[key]['s0']:.4f} "
                  f"CRPS={results[key]['crps']:.4f} sd={sd:.4f}", flush=True)

    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(
        {"model": a.model, "n": len(rows), "results": results}, indent=2))

    pick_s0 = max(results, key=lambda k: results[k]["s0"])
    pick_cr = min(results, key=lambda k: results[k]["crps"])
    print(f"\nS_0 would select   {pick_s0}  (sd={results[pick_s0]['sd']:.4f})")
    print(f"S_1/2 would select {pick_cr}  (sd={results[pick_cr]['sd']:.4f})")
    print("REACHABLE: S_0 prefers a lower-dispersion prompt"
          if results[pick_s0]["sd"] < results[pick_cr]["sd"] - 1e-6
          else "NOT SEPARATED: S_0's pick is not less dispersed")


if __name__ == "__main__":
    main()
