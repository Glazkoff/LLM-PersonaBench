"""Probe why a model's Likert belief readout returns no mass on {1..5}."""
import math, os, sys
from openai import OpenAI

model = sys.argv[1]
c = OpenAI(api_key="EMPTY", base_url=os.environ["LOCAL_LLM_BASE_URL"])
SYS = "You are simulating a person answering a personality questionnaire."
Q = ("Worry about things.\nAnswer with a single digit from 1 to 5. "
     "Reply with the digit only.")

def top(msgs, **extra):
    r = c.chat.completions.create(model=model, messages=msgs, max_tokens=1,
                                  temperature=0.0, logprobs=True, top_logprobs=10, **extra)
    lp = r.choices[0].logprobs.content[0].top_logprobs
    return [(t.token, round(math.exp(t.logprob), 4)) for t in lp]

print("A) assistant-prefill + thinking off:")
try:
    print("  ", top([{"role":"system","content":SYS},{"role":"user","content":Q},
                     {"role":"assistant","content":"My answer is "}],
                    extra_body={"chat_template_kwargs":{"enable_thinking":False},
                                "add_generation_prompt":False,"continue_final_message":True}))
except Exception as e:
    print("   FAILED:", repr(e)[:200])

print("B) plain chat, no prefill:")
try:
    print("  ", top([{"role":"system","content":SYS},{"role":"user","content":Q}]))
except Exception as e:
    print("   FAILED:", repr(e)[:200])

print("C) free generation (what does it actually say?):")
try:
    r = c.chat.completions.create(model=model,
        messages=[{"role":"system","content":SYS},{"role":"user","content":Q}],
        max_tokens=24, temperature=0.0)
    print("   ", repr(r.choices[0].message.content))
except Exception as e:
    print("   FAILED:", repr(e)[:200])

print("D) completions endpoint with raw prompt:")
try:
    r = c.completions.create(model=model, prompt=f"{SYS}\n\n{Q}\nAnswer: ",
                             max_tokens=1, temperature=0.0, logprobs=10)
    lg = r.choices[0].logprobs
    print("   ", lg.top_logprobs[0] if lg and lg.top_logprobs else "(none)")
except Exception as e:
    print("   FAILED:", repr(e)[:200])
