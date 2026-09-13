r"""H18: did optimising $S_0$ make the simulator better or worse?

The companion method paper evolved cluster-level system prompts to maximise
answer similarity $S_0$. This paper proves $S_0$ is improper and shows it
selects badly among models. The sharper question is whether OPTIMISING it
degraded the artefact that was optimised.

So: take the evolved genotypes and the base genotypes those runs started from,
build system prompts with the method paper's own assembly code, and read the
same belief distribution under both. Score with the objective that was
optimised ($S_0$) and with proper rules ($S_{1/2}$, log-loss).

The prediction the audit makes is specific and falsifiable: $S_0$ should
improve, because that is what was maximised, while the proper scores should
not -- and may worsen. If instead both improve, the audit's practical bite is
much weaker and this script says so.

Respondents are drawn fresh from each cluster. The evolution used 100 users per
cluster out of 410,168, so overlap is negligible, but these are not certified
disjoint from its training split and we do not claim they are.
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


def load_system():
    return json.loads((ROOT / "src/prompt/system.json").read_text())


def load_base_genotypes(sysmsg):
    tr = json.loads((ROOT / "src/prompt/previous_base_prompt/traits.json").read_text())
    fc = json.loads((ROOT / "src/prompt/previous_base_prompt/facets.json").read_text())
    out = {}
    for c in tr:
        out[int(c)] = {
            "role_definition": sysmsg["role"],
            "trait_formulations": tr[c],
            "facet_formulations": fc.get(c, {}),
            "intensity_modifiers": sysmsg["intensity_modifiers"],
            "critic_formulations": sysmsg["critic_internal"],
            "template_structure": sysmsg["template_structure"],
        }
    return out


def load_evolved_genotypes(sysmsg):
    """The evolved file references `system` at module scope; supply it."""
    src = (ROOT / "src/prompt/best_genotype_giga_evoprompt_iter1"
                  "/gigachat3_best_genotype_evoprompt.py").read_text()
    ns = {"system": sysmsg}
    exec(compile(src, "evolved", "exec"), ns)
    return {int(k.rsplit("_", 1)[1]): v for k, v in ns.items()
            if k.startswith("best_genotype_")}


def system_prompt(gen, row):
    """The method paper's assembly, minus the all-120-questions user turn."""
    mods = gen["intensity_modifiers"]
    tt, ft = gen.get("trait_targets") or {}, gen.get("facet_targets") or {}
    lines = []
    for trait, desc in gen["trait_formulations"].items():
        v = row.get(trait)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        t = tt.get(trait)
        m = get_modifier_by_match(v, t, mods) if t is not None else get_modifier_bisect(v, mods)
        lines.append(f"- This trait ({trait}) describes you {m}: {desc}")
    traits_text = "\n".join(lines)
    lines = []
    for facet, desc in gen["facet_formulations"].items():
        v = row.get(facet)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        t = ft.get(facet)
        m = get_modifier_by_match(v, t, mods) if t is not None else get_modifier_bisect(v, mods)
        lines.append(f"- This facet ({facet}) describes you {m}: {desc}")
    facets_text = "\n".join(lines)
    return (f"{gen['role_definition']}\n"
            f"        Your traits:\n        {traits_text}\n"
            f"        Your specific behavioral aspects:\n        {facets_text}\n"
            f"\n        Internal reflection guideline:\n"
            f"        {gen['critic_formulations']}\n    ")


def crps(Q, Y):
    ind = np.stack([(Y <= t).astype(float) for t in THRESH], axis=2)
    return np.where(np.isnan(Y), np.nan, np.nansum((Q - ind) ** 2, axis=2) / 4.0)


def s0(P, Y):
    lv = np.arange(1, 6, dtype=float)
    d = np.abs(lv[None, None, :] - Y[:, :, None]) / 4.0
    return np.where(np.isnan(Y), np.nan, np.nansum(P * (1.0 - d), axis=2))


def logloss(P, Y):
    idx = np.clip(np.nan_to_num(Y, nan=1).astype(int) - 1, 0, 4)
    p = np.take_along_axis(P, idx[:, :, None], axis=2)[:, :, 0]
    return np.where(np.isnan(Y), np.nan,
                    np.where(p > 0, -np.log(np.clip(p, 1e-300, None)), np.inf))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-cluster", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=260913)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sysmsg = load_system()
    base, evolved = load_base_genotypes(sysmsg), load_evolved_genotypes(sysmsg)
    common = sorted(set(base) & set(evolved))
    print(f"clusters with both genotypes: {common}", flush=True)

    df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
    # Reuse the study's own loader rather than re-deriving item text, reverse
    # keys and storage orientation. The answers live in columns i1..i120 (not
    # 1..120), and this corpus stores them ALREADY reverse-recoded while the
    # model answers the literal item -- an orientation mismatch that silently
    # invalidated an earlier round of this project.
    corpus = load_corpus("ipip")
    items = [{"id": i, "text": corpus.text[i]} for i in corpus.ids]
    Yall = corpus.Y
    flip = np.array([i in corpus.rev for i in corpus.ids], dtype=bool)
    print(f"items={len(items)} reverse-keyed={int(flip.sum())} "
          f"corpus_recoded={corpus.recoded}", flush=True)

    rng = np.random.default_rng(a.seed)
    rows = []
    for c in common:
        pool = df.index[df["clusters"] == c].to_numpy()
        take = rng.choice(pool, min(a.per_cluster, len(pool)), replace=False)
        rows += [(c, i) for i in take]
    print(f"respondents: {len(rows)} across {len(common)} clusters", flush=True)

    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True)
    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=dtype, device_map="auto", trust_remote_code=True).eval()
    dids = [tok.encode(d, add_special_tokens=False)[0] for d in DIGITS]
    tmpl = getattr(tok, "chat_template", "") or ""
    prefill = ("<|channel|>final<|message|>My answer is "
               if "<|channel|>" in tmpl else "My answer is ")

    out = {}
    for cond, gens in (("base", base), ("evolved", evolved)):
        prompts, Y = [], np.full((len(rows), len(items)), np.nan)
        for r, (c, idx) in enumerate(rows):
            row = df.loc[idx]
            sysp = system_prompt(gens[c], row)
            for j, it in enumerate(items):
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
            Y[r] = Yall[idx]

        P = np.full((len(rows), len(items), 5), np.nan)
        mass = []
        with torch.no_grad():
            for b0 in range(0, len(prompts), a.batch_size):
                chunk = prompts[b0:b0 + a.batch_size]
                enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
                pr = torch.softmax(model(**enc).logits[:, -1, :].float(), -1)[:, dids]
                pr = pr.cpu().numpy()
                tot = pr.sum(1, keepdims=True)
                mass.append(tot.ravel())
                for k in range(len(chunk)):
                    g = b0 + k
                    P[g // len(items), g % len(items)] = pr[k] / max(tot[k, 0], 1e-9)
                if b0 % (a.batch_size * 200) == 0:
                    print(f"  {cond} {b0}/{len(prompts)}", flush=True)

        # The model answered the literal item; the corpus stores recoded values.
        # Flip the model's simplex on reverse-keyed items so both sides share an
        # orientation. Variance is invariant to this, absolute-error scores are not.
        if corpus.recoded and flip.any():
            P[:, flip, :] = P[:, flip, ::-1]
        Q = np.cumsum(P, axis=2)[:, :, :4]
        out[cond] = dict(
            s0=float(np.nanmean(s0(P, Y))),
            crps=float(np.nanmean(crps(Q, Y))),
            logloss=float(np.nanmean(logloss(P, Y))),
            mass=float(np.nanmean(np.concatenate(mass))),
            per_resp_s0=np.nanmean(s0(P, Y), axis=1).tolist(),
            per_resp_crps=np.nanmean(crps(Q, Y), axis=1).tolist(),
            per_resp_ll=np.nanmean(logloss(P, Y), axis=1).tolist(),
        )
        print(f"{cond}: S0={out[cond]['s0']:.4f} CRPS={out[cond]['crps']:.4f} "
              f"logloss={out[cond]['logloss']:.4f} mass={out[cond]['mass']:.4f}", flush=True)

    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if not kk.startswith("per_resp")}
         for k, v in out.items()} | {"model": a.model, "n": len(rows),
                                     "clusters": common}, indent=2))
    np.savez(d / "per_respondent.npz", **{
        f"{c}_{m}": np.array(out[c][f"per_resp_{m}"]) for c in out for m in ("s0", "crps", "ll")})

    rng2 = np.random.default_rng(a.seed + 1)
    print("\npaired evolved - base, bootstrap over respondents:")
    for m, lab, better in (("s0", "S_0 (optimised)", "higher"),
                           ("crps", "S_1/2", "lower"),
                           ("ll", "log-loss", "lower")):
        dv = np.array(out["evolved"][f"per_resp_{m}"]) - np.array(out["base"][f"per_resp_{m}"])
        bs = np.percentile([dv[rng2.integers(0, len(dv), len(dv))].mean()
                            for _ in range(4000)], [2.5, 97.5])
        verdict = "resolved" if bs[0] * bs[1] > 0 else "unresolved"
        direction = ("improved" if ((dv.mean() > 0) == (better == "higher")) else "WORSENED")
        print(f"  {lab:16s} {dv.mean():+.4f} [{bs[0]:+.4f},{bs[1]:+.4f}] "
              f"{verdict:10s} {direction if verdict=='resolved' else ''}")


if __name__ == "__main__":
    main()
