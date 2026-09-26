"""Readout probe for one served model: mass on the answer tokens and a smoke fitness evaluation."""
import argparse
import json
import sys
import time

import numpy as np

from src.optim.genotype import seed_genotype
from src.optim.panels import ITEM_SPLIT, answers, load_corpus, persona_scores
from src.optim.persona import system_prompt
from src.optim.readout import VLLMReadout, render_prompt, ITEM_QUESTION, TEXT_OF_ITEM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefill", default="My answer is ")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = VLLMReadout(a.model, prefill=a.prefill)
    df = load_corpus()
    rows = df[df["clusters"] == 0].head(a.n)
    systems = [system_prompt(seed_genotype(0), persona_scores(x)) for x in answers(rows, ITEM_SPLIT[0])]
    print("PROMPT SAMPLE:\n" + render_prompt(r.tok, systems[0], ITEM_QUESTION.format(text=TEXT_OF_ITEM[ITEM_SPLIT[1][0]]),
                                             a.prefill)[-600:], flush=True)
    one = render_prompt(r.tok, systems[0], ITEM_QUESTION.format(text=TEXT_OF_ITEM[ITEM_SPLIT[1][0]]), a.prefill)
    dbg = r.client.completions.create(model=a.model, prompt=[r._ids(one)], max_tokens=8, temperature=0.0, logprobs=20)
    ch = dbg.choices[0]
    print("DEBUG continuation:", repr(ch.text), flush=True)
    print("DEBUG first-token top-20:", ch.logprobs.top_logprobs[0] if ch.logprobs and ch.logprobs.top_logprobs else None, flush=True)
    print("DEBUG prompt token count:", len(r.tok(one, add_special_tokens=False)["input_ids"]), flush=True)
    t0 = time.time()
    P, M, tok = r.beliefs(systems, ITEM_SPLIT[1])
    dt = time.time() - t0
    res = {"model": a.model, "mass_mean": float(M.mean()), "mass_min": float(M.min()), "n_requests": int(M.size),
           "seconds": dt, "req_per_s": M.size / dt, "mean_belief": P.mean((0, 1)).round(3).tolist()}
    print(json.dumps(res), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    sys.exit(0 if res["mass_mean"] >= 0.5 else 2)


if __name__ == "__main__":
    main()
