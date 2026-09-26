"""Belief readout: the model's probability over the five answer digits for each (persona, item).

Prompts are rendered with the model's own chat template (thinking disabled) and the assistant turn is
prefilled, then sent to vLLM's /v1/completions endpoint in batches with max_tokens=1 and top-20
logprobs. Rendering locally keeps the token sequence identical to an HF forward pass; the audit found
that relying on the server's chat endpoint to apply prefills lost the answer-token mass for some models.
"""
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.optim.panels import TEXT_OF_ITEM

DIGITS = ["1", "2", "3", "4", "5"]
ITEM_QUESTION = ('How accurately does this statement describe you? "{text}"\n'
                 "Answer with a single number from 1 (very inaccurate) to 5 (very accurate).")


class FakeReadout:
    def __init__(self, rng):
        self.rng = rng

    def beliefs(self, systems, item_ids, texts=None):
        n, J = len(systems), len(item_ids)
        return self.rng.dirichlet(np.ones(5), size=(n, J)), np.full((n, J), 0.99), 100 * n * J


def render_prompt(tok, system, user, prefill):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kw = dict(tokenize=False, add_generation_prompt=True)
    try:
        s = tok.apply_chat_template(msgs, enable_thinking=False, **kw)
    except Exception:
        try:
            s = tok.apply_chat_template(msgs, **kw)
        except Exception:
            s = tok.apply_chat_template([{"role": "user", "content": system + "\n\n" + user}], **kw)
    # some templates open an empty think block when thinking is disabled; keep it -- it is what the model expects
    return s + prefill


class VLLMReadout:
    def __init__(self, model, base_url=None, prefill="My answer is ", batch=64, workers=16, tokenizer=None):
        from openai import OpenAI
        from transformers import AutoTokenizer
        self.model, self.prefill, self.batch, self.workers = model, prefill, batch, workers
        self.tok = tokenizer or AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.client = OpenAI(base_url=base_url or os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
                             api_key=os.environ.get("LOCAL_LLM_API_KEY", "EMPTY"), timeout=1800, max_retries=3)

    def _batch(self, prompts):
        r = self.client.completions.create(model=self.model, prompt=prompts, max_tokens=1, temperature=0.0,
                                           logprobs=20)
        out = [None] * len(prompts)
        for ch in r.choices:
            top = (ch.logprobs.top_logprobs or [{}])[0] or {}
            p = np.zeros(5)
            for t, lp in top.items():
                t = t.strip()
                if t in DIGITS:
                    p[int(t) - 1] += math.exp(lp)
            out[ch.index] = p
        return out, int(getattr(r.usage, "prompt_tokens", 0) or 0)

    def beliefs(self, systems, item_ids, texts=None):
        """Two phases so vLLM's prefix cache pays off: (1) one item per respondent, which computes and caches
        each persona's system-prompt blocks; (2) every respondent's remaining items as one request, which then
        only prefills the item question."""
        n, J = len(systems), len(item_ids)
        texts = texts or TEXT_OF_ITEM
        prompts = [[render_prompt(self.tok, systems[a], ITEM_QUESTION.format(text=texts[item_ids[b]]), self.prefill)
                    for b in range(J)] for a in range(n)]
        warm = [[prompts[a][0] for a in range(i, min(i + self.batch, n))] for i in range(0, n, self.batch)]
        with ThreadPoolExecutor(self.workers) as ex:
            res_w = list(ex.map(self._batch, warm))
            res_r = list(ex.map(self._batch, [prompts[a][1:] for a in range(n)])) if J > 1 else []
        first = [p for ps, _ in res_w for p in ps]
        rows = [[first[a]] + (res_r[a][0] if J > 1 else []) for a in range(n)]
        tok = sum(t for _, t in res_w) + sum(t for _, t in res_r)
        raw = np.array(rows).reshape(n, J, 5)
        mass = raw.sum(-1)
        P = np.where(mass[..., None] > 0, raw / np.maximum(mass[..., None], 1e-12), 0.2)
        return P, mass, tok
