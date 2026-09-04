"""
Local vLLM provider (OpenAI-compatible), for running open-weight models on the
AIRI/Euler cluster instead of a paid API.

Two capabilities the cloud providers in this repo do not offer:

1. Real concurrency. `generate_batch` issues requests in a thread pool, so a
   local vLLM server is driven at its actual throughput rather than one
   request at a time.

2. `likert_distribution()` -- the token-level distribution over the answer
   options {1,2,3,4,5} for a single item, read straight from the model's
   logprobs. This makes it possible to separate two very different causes of
   under-dispersed simulated respondents:
       (a) the model's own answer distribution is already narrow, versus
       (b) the distribution is wide but greedy/low-temperature decoding
           collapses it to the argmax.
   Sampling from (a) costs one forward pass and no sampling noise.
"""
import os
from concurrent.futures import ThreadPoolExecutor

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI

from src.models.base import BaseLLM

DEFAULT_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
LIKERT_TOKENS = ["1", "2", "3", "4", "5"]


class LocalVLLMModel(BaseLLM):
    def __init__(self, model_name: str, temperature: float = 0.7,
                 timeout: float | None = None, max_retries: int | None = None,
                 base_url: str | None = None, max_workers: int = 32,
                 top_p: float | None = None, seed: int | None = None):
        super().__init__(model_name)
        self.base_url = base_url or DEFAULT_BASE_URL
        self.api_key = os.environ.get("LOCAL_LLM_API_KEY", "EMPTY")
        self.max_workers = max_workers
        self.temperature = temperature

        kwargs: dict = {"model": model_name, "api_key": self.api_key,
                        "base_url": self.base_url, "temperature": temperature}
        if timeout is not None:
            kwargs["timeout"] = timeout
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        extra = {}
        if top_p is not None:
            extra["top_p"] = top_p
        if seed is not None:
            extra["seed"] = seed
        if extra:
            kwargs["model_kwargs"] = {"extra_body": extra}
        self.llm = ChatOpenAI(**kwargs)
        # raw client for logprob access, which LangChain does not expose cleanly
        self._raw = OpenAI(api_key=self.api_key, base_url=self.base_url,
                           timeout=timeout or 600.0)

    # ------------------------------------------------------------ generation
    def generate(self, prompt) -> str:
        return (prompt | self.llm).invoke({})

    def generate_batch(self, prompts: list) -> list:
        """Concurrent batch. Order is preserved; failures return None."""
        if not prompts:
            return []

        def one(p):
            try:
                return self.generate(p)
            except Exception as e:  # a single bad request must not kill a sweep
                return RuntimeError(f"local_vllm request failed: {e!r}")

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(prompts))) as ex:
            return list(ex.map(one, prompts))

    # ------------------------------------------------- token-level readout
    def likert_distribution(self, system_prompt: str, question: str,
                            options: list[str] | None = None) -> dict:
        """
        Return the model's own probability distribution over the Likert options
        for ONE item, from the logprobs of the first generated token.

        Returns {"probs": {"1": p1, ..., "5": p5}, "argmax": int,
                 "expected": float, "entropy": float, "mass_on_scale": float}
        `mass_on_scale` is how much probability landed on the five valid answer
        tokens before renormalisation -- a low value means the model wanted to
        say something else, and the renormalised distribution is unreliable.
        """
        import math

        options = options or LIKERT_TOKENS
        resp = self._raw.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": question}],
            max_tokens=1, temperature=0.0, logprobs=True, top_logprobs=20,
        )
        top = resp.choices[0].logprobs.content[0].top_logprobs
        raw = {}
        total = 0.0
        for t in top:
            tok = t.token.strip()
            p = math.exp(t.logprob)
            total += p
            if tok in options:
                raw[tok] = raw.get(tok, 0.0) + p

        on_scale = sum(raw.values())
        if on_scale <= 0:
            return {"probs": None, "argmax": None, "expected": None,
                    "entropy": None, "mass_on_scale": 0.0}
        probs = {o: raw.get(o, 0.0) / on_scale for o in options}
        exp_val = sum(float(o) * p for o, p in probs.items())
        ent = -sum(p * math.log(p) for p in probs.values() if p > 0)
        return {"probs": probs,
                "argmax": int(max(probs, key=probs.get)),
                "expected": exp_val,
                "entropy": ent,
                "mass_on_scale": on_scale / max(total, 1e-12)}

    def info(self) -> str:
        return f"LocalVLLM({self.model_name} @ {self.base_url})"
