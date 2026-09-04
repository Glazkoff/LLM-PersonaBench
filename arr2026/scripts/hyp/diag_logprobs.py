"""Tiny diagnostic: what does the server actually return for one Likert item?"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from src.models.providers.local_vllm import LocalVLLMModel
from openai import OpenAI

model = sys.argv[1]
base = os.environ["LOCAL_LLM_BASE_URL"]
sysmsg = "You are simulating a person answering a personality questionnaire."
q = ("Worry about things.\nAnswer with a single digit from 1 to 5. Reply with the digit only.")

print("--- raw chat, no tricks ---", flush=True)
c = OpenAI(api_key="EMPTY", base_url=base)
r = c.chat.completions.create(model=model,
    messages=[{"role":"system","content":sysmsg},{"role":"user","content":q}],
    max_tokens=8, temperature=0.0, logprobs=True, top_logprobs=8)
print("content:", repr(r.choices[0].message.content))
print("first-token top:", [(t.token, round(t.logprob,2)) for t in r.choices[0].logprobs.content[0].top_logprobs[:8]])

print("\n--- provider.likert_distribution ---", flush=True)
m = LocalVLLMModel(model, base_url=base, timeout=120)
d = m.likert_distribution(sysmsg, q)
print("result:", d)
print("top tokens seen:", getattr(m, "last_top_tokens", None))
