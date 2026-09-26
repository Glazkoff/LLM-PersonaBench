import math


class BudgetExhausted(RuntimeError):
    pass


class BudgetLedger:
    """B candidate evaluations. A full-panel evaluation charges 1; a minibatch charges its fraction."""

    def __init__(self, B: int):
        self.B = int(B)
        self.used_f = 0.0
        self.forward_passes = 0
        self.prompt_tokens = 0
        self.generated_tokens = 0

    @property
    def used(self) -> int:
        return math.ceil(self.used_f - 1e-9)

    @property
    def remaining(self) -> int:
        return max(0, self.B - self.used)

    def remaining_fraction(self) -> float:
        return max(0.0, self.B - self.used_f)

    def charge_fraction(self, frac, passes=0, prompt_tokens=0, generated_tokens=0):
        if self.used_f + frac > self.B + 1e-9:
            raise BudgetExhausted(f"B={self.B} exhausted")
        self.used_f += frac
        self.forward_passes += int(passes)
        self.prompt_tokens += int(prompt_tokens)
        self.generated_tokens += int(generated_tokens)

    def charge(self, passes, prompt_tokens=0, generated_tokens=0):
        self.charge_fraction(1.0, passes, prompt_tokens, generated_tokens)

    def to_dict(self):
        return {"budget_B": self.B, "candidate_evaluations": self.used,
                "candidate_evaluations_exact": round(self.used_f, 4), "forward_passes": self.forward_passes,
                "prompt_tokens": self.prompt_tokens, "generated_tokens": self.generated_tokens}
