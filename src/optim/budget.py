"""Search budget measured in executed respondent-item forecasts."""

import math
import threading


class BudgetExhausted(RuntimeError):
    pass


class BudgetLedger:
    def __init__(self, B: int, full_panel_forecasts: int | None = None):
        self.B = int(B)
        self.full_panel_forecasts = full_panel_forecasts
        self.executed_forecasts = 0
        self.forward_passes = 0  # legacy name for executed forecasts
        self.prompt_tokens = 0
        self.generated_tokens = 0
        self.proposal_attempts = 0
        self.invalid_proposals = 0
        self.mutator_calls = 0
        self.hypermutation_calls = 0
        self.evaluator_calls = 0
        self._lock = threading.Lock()

    def configure(self, full_panel_forecasts: int):
        n = int(full_panel_forecasts)
        if n <= 0 or (self.full_panel_forecasts is not None and self.full_panel_forecasts != n):
            raise ValueError("inconsistent full-panel forecast count")
        self.full_panel_forecasts = n

    @property
    def used_f(self):
        return self.executed_forecasts / self.full_panel_forecasts if self.full_panel_forecasts else 0.0

    @property
    def used(self):
        return math.ceil(self.used_f - 1e-9)

    @property
    def remaining(self):
        return max(0, self.B - self.used)

    def remaining_fraction(self):
        return max(0.0, self.B - self.used_f)

    def can_charge(self, forecasts: int):
        if self.full_panel_forecasts is None:
            raise ValueError("full-panel forecast count is not configured")
        if forecasts < 0:
            raise ValueError("forecast count must be nonnegative")
        return self.executed_forecasts + forecasts <= self.B * self.full_panel_forecasts

    def charge_forecasts(self, forecasts: int, prompt_tokens=0, generated_tokens=0):
        if int(forecasts) != forecasts or forecasts < 0:
            raise ValueError("forecast count must be a nonnegative integer")
        with self._lock:
            if not self.can_charge(forecasts):
                raise BudgetExhausted(f"B={self.B} exhausted")
            self.executed_forecasts += int(forecasts)
            self.forward_passes = self.executed_forecasts
            self.evaluator_calls += int(forecasts > 0)
            self.prompt_tokens += int(prompt_tokens)
            self.generated_tokens += int(generated_tokens)

    def record_proposal(self, valid=True, prompt_tokens=0, generated_tokens=0, hypermutation=False):
        self.proposal_attempts += 1
        self.invalid_proposals += int(not valid)
        self.mutator_calls += 1
        self.hypermutation_calls += int(hypermutation)
        self.prompt_tokens += int(prompt_tokens)
        self.generated_tokens += int(generated_tokens)

    def charge_fraction(self, frac, passes=0, prompt_tokens=0, generated_tokens=0):
        # Compatibility with legacy arms. A fraction alone is never a forecast.
        if passes == 0 and frac:
            raise ValueError("cannot charge an evaluation without executed forecasts")
        if self.full_panel_forecasts is not None and passes and not math.isclose(
                passes / self.full_panel_forecasts, frac, rel_tol=0, abs_tol=1e-9):
            raise ValueError("fraction does not match executed forecasts")
        self.charge_forecasts(passes, prompt_tokens, generated_tokens)

    def charge(self, passes, prompt_tokens=0, generated_tokens=0):
        self.charge_forecasts(passes, prompt_tokens, generated_tokens)

    def to_dict(self):
        return {"budget_B": self.B, "full_panel_forecasts": self.full_panel_forecasts,
                "executed_forecasts": self.executed_forecasts, "full_panel_equivalents": self.used_f,
                "candidate_evaluations": self.used, "candidate_evaluations_exact": self.used_f,
                "forward_passes": self.forward_passes, "evaluator_calls": self.evaluator_calls,
                "proposal_attempts": self.proposal_attempts, "invalid_proposals": self.invalid_proposals,
                "mutator_calls": self.mutator_calls, "hypermutation_calls": self.hypermutation_calls,
                "prompt_tokens": self.prompt_tokens, "generated_tokens": self.generated_tokens,
                "gepa_total_metric_calls": getattr(self, "gepa_total_metric_calls", None)}
