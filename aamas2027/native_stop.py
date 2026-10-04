"""External forecast cap composed with the native GigaEvo mutant stopper."""
from pathlib import Path
from gigaevo.evolution.engine.stopper import MaxMutantsStopper, StopDecision


class ForecastStopper(MaxMutantsStopper):
    def __init__(self, max_mutants, flag):
        super().__init__(max_mutants)
        self.flag = Path(flag)

    def should_stop(self, ctx):
        if self.flag.exists():
            return StopDecision(True, 'external forecast budget reached', 'forecast_budget')
        return super().should_stop(ctx)

    def remaining_dispatches(self, ctx):
        return 0 if self.flag.exists() else super().remaining_dispatches(ctx)
