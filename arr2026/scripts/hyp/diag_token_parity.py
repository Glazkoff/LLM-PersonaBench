"""
Token-context parity audit (idea 2).

The 2026 reasoning-first models cannot be read through a chat prefill: they open
with a reasoning preamble and place ~0.08% of mass on the answer tokens. The raw
completions endpoint reads them cleanly. The question this settles is whether
switching endpoints is a TRANSPORT change or an ELICITATION change.

If the completions endpoint can be handed the byte-identical token sequence the
chat template would produce, then the two paths are the same conditioning and the
already-measured models need no re-run. If not, the 2026 models must be reported
as a protocol-stratified cohort.

Test: serialise the chat template locally, send those exact tokens to
/v1/completions, and compare the next-token distribution against /v1/chat.
"""
import json, math, os, sys
import numpy as np
from openai import OpenAI
from transformers import AutoTokenizer

model = sys.argv[1]
base = os.environ["LOCAL_LLM_BASE_URL"]
c = OpenAI(api_key="EMPTY", base_url=base)
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
DIG = ["1","2","3","4","5"]

SYS = "You are simulating a person answering a personality questionnaire."
QS = ["Worry about things.", "Love large parties.", "Believe in the importance of art.",
      "Trust others.", "Complete tasks successfully.", "Get angry easily."]
INSTR = "Answer with a single digit from 1 to 5. Reply with the digit only."

def dist_from_top(tl):
    raw = {t.token.strip(): math.exp(t.logprob) for t in tl}
    p = np.array([raw.get(d, 0.0) for d in DIG]); m = p.sum()
    return (p/m if m > 0 else p*0), m

rows = []
errs = []
for q in QS:
    msgs = [{"role":"system","content":SYS},{"role":"user","content":f"{q}\n{INSTR}"}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prefill = text + "My answer is"          # NOTE: no trailing space

    # A) chat endpoint with assistant prefill
    try:
        r = c.chat.completions.create(model=model, messages=msgs+[{"role":"assistant","content":"My answer is"}],
            max_tokens=1, temperature=0.0, logprobs=True, top_logprobs=20,
            extra_body={"chat_template_kwargs":{"enable_thinking":False},
                        "add_generation_prompt":False,"continue_final_message":True})
        pa, ma = dist_from_top(r.choices[0].logprobs.content[0].top_logprobs)
    except Exception as e:
        print(f"    CHAT FAILED: {e!r}"[:300], flush=True)
        errs.append(("chat", repr(e)[:200]))
        pa, ma = np.zeros(5), 0.0

    # B) completions endpoint with the SAME serialised text
    try:
        r2 = c.completions.create(model=model, prompt=prefill, max_tokens=1,
                                  temperature=0.0, logprobs=20)
        tl = r2.choices[0].logprobs.top_logprobs[0]
        raw = {k.strip(): math.exp(v) for k, v in tl.items()}
        pb = np.array([raw.get(d,0.0) for d in DIG]); mb = pb.sum()
        pb = pb/mb if mb > 0 else pb
    except Exception as e:
        print(f"    COMPLETIONS FAILED: {e!r}"[:300], flush=True)
        errs.append(("completions", repr(e)[:200]))
        pb, mb = np.zeros(5), 0.0

    # Jensen-Shannon between the two renormalised answer distributions
    def js(p, q):
        if p.sum() == 0 or q.sum() == 0: return float("nan")
        m = 0.5*(p+q)
        kl = lambda a,b: np.sum(np.where(a>0, a*np.log(a/np.where(b>0,b,1e-12)), 0))
        return 0.5*kl(p,m)+0.5*kl(q,m)

    rows.append({"item": q[:28], "mass_chat": round(float(ma),5), "mass_compl": round(float(mb),5),
                 "js": round(float(js(pa,pb)),5),
                 "mean_chat": round(float((pa*np.arange(1,6)).sum()),3),
                 "mean_compl": round(float((pb*np.arange(1,6)).sum()),3)})
    print(f"  {q[:28]:<30} mass chat={ma:.4f} compl={mb:.4f}  JS={rows[-1]['js']:.4f} "
          f"mean {rows[-1]['mean_chat']:.2f} vs {rows[-1]['mean_compl']:.2f}", flush=True)

ok = [r for r in rows if r["mass_compl"] > 0.1]
js_med = float(np.nanmedian([r["js"] for r in ok])) if ok else float("nan")
res = {"model": model, "n_items": len(rows), "n_completions_usable": len(ok),
       "median_JS_chat_vs_completions": None if math.isnan(js_med) else round(js_med,5),
       "mean_mass_chat": round(float(np.mean([r["mass_chat"] for r in rows])),5),
       "mean_mass_completions": round(float(np.mean([r["mass_compl"] for r in rows])),5),
       "errors": errs[:6]}
if res["mean_mass_chat"] < 0.05:
    res["valid"] = False
    res["verdict"] = ("INVALID: the chat path itself returned no mass on the answer tokens, "
                      "so this run measures a harness failure, not endpoint equivalence. "
                      "Do NOT read a verdict from it.")
else:
    res["valid"] = True
    res["verdict"] = ("TRANSPORT-EQUIVALENT: same tokens, same distribution; endpoints interchangeable"
                      if (not math.isnan(js_med) and js_med < 0.01) else
                      "NOT EQUIVALENT: report the completions cohort separately")
print("\n=== PARITY SUMMARY ==="); print(json.dumps(res, indent=2))
