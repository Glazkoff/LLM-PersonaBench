"""Bounded localhost load for a colocated mutator serving smoke; no result scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--requests", type=int, default=120)
    a = ap.parse_args()
    client = OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=120, max_retries=1)

    def send(i):
        response = client.completions.create(
            model=a.model,
            prompt=f"Rewrite this fictional survey persona in one sentence. Variant {i}: a careful, curious respondent.",
            max_tokens=128,
            temperature=0.7,
        )
        return bool(response.choices)

    with ThreadPoolExecutor(max_workers=4) as pool:
        ok = sum(pool.map(send, range(a.requests)))
    print(f"mutator requests succeeded: {ok}/{a.requests}", flush=True)
    if ok != a.requests:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
