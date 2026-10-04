import json

import numpy as np
import pytest

from src.optim.arms import ARMS
from src.optim.budget import BudgetExhausted, BudgetLedger
from src.optim.results import CellWriter
from tests.optim.conftest import TEMPLATE


class TinyFitness:
    n_opt = 4
    tgt = (1, 2)
    forward_passes_per_candidate = 8
    last_prompt_tokens = 3

    def evaluate(self, genotype):
        from src.optim.fitness import FitnessResult
        return FitnessResult(0.2, 0.2, 0.2, 1.0, 3, 0.0, (0.0,))


class BadMutator:
    def complete(self, *args, **kwargs):
        return "bad JSON", 7, 5


def test_forecast_ledger_rejects_overspend_and_does_not_charge_invalid_attempt():
    ledger = BudgetLedger(2, full_panel_forecasts=8)
    ledger.charge_forecasts(3, prompt_tokens=4)
    ledger.record_proposal(valid=False, prompt_tokens=7, generated_tokens=5)
    assert ledger.executed_forecasts == 3
    assert ledger.used_f == 3 / 8
    assert ledger.invalid_proposals == 1
    assert ledger.prompt_tokens == 11
    with pytest.raises(BudgetExhausted):
        ledger.charge_forecasts(14)
    assert ledger.executed_forecasts == 3


def test_invalid_proposals_terminate_without_phantom_forecasts(tmp_path):
    ledger = BudgetLedger(2)
    writer = CellWriter(tmp_path, "random")
    arm = ARMS["random"]({"max_invalid_proposals": 3}, np.random.default_rng(1), BadMutator(), TEMPLATE)
    chosen = arm.run(TinyFitness(), TEMPLATE, ledger, writer)
    assert chosen == TEMPLATE
    assert ledger.executed_forecasts == 8
    assert ledger.invalid_proposals == 3
    assert ledger.proposal_attempts == 3
    assert ledger.used_f == 1
    selected = json.loads((writer.dir / "selected_genotype.json").read_text())
    assert selected["genotype"] == chosen
    assert selected["candidate_id"] == arm.best_candidate_id


class MetaMutator:
    def complete(self, system, user, **kwargs):
        if "Write a different instruction" in user:
            return "Change the phrasing", 11, 13
        return json.dumps(TEMPLATE), 5, 6


def test_promptbreeder_accounts_hypermutation_tokens(tmp_path):
    ledger = BudgetLedger(3)
    writer = CellWriter(tmp_path, "pb")
    arm = ARMS["promptbreeder"]({"population_size": 2, "hyper_mutation_rate": 1.0},
                                  np.random.default_rng(1), MetaMutator(), TEMPLATE)
    arm.run(TinyFitness(), TEMPLATE, ledger, writer)
    assert ledger.executed_forecasts == 24
    assert ledger.prompt_tokens >= 11
    assert ledger.generated_tokens >= 13
    assert any(row.get("hypermutation_prompt_tokens") == 11 for row in arm.history)


def test_native_gepa_executes_official_engine_on_fixed_opt_panel(tmp_path):
    pytest.importorskip("gepa")
    from src.optim.arms.gepa_official import OfficialGEPA

    class Fitness:
        n_opt = 3
        tgt = (1, 2)
        forward_passes_per_candidate = 6
        Y = np.ones((3, 2), dtype=int)
        last_prompt_tokens = 2

        def _beliefs(self, g, idx):
            P = np.zeros((len(idx), 2, 5))
            P[:, :, 0 if "better" in g["role_definition"] else 4] = 1
            return P, np.ones((len(idx), 2))

    class Reflection:
        def complete(self, system, user, **kw):
            better = dict(TEMPLATE)
            better["role_definition"] = "better"
            return "```json\n" + json.dumps({k: better[k] for k in (
                "role_definition", "trait_formulations", "facet_formulations", "critic_formulations")}) + "\n```", 9, 10

    writer = CellWriter(tmp_path, "gepa_official")
    ledger = BudgetLedger(5)
    arm = OfficialGEPA({"minibatch_respondents": 1, "seed": 3}, np.random.default_rng(3), Reflection(), TEMPLATE)
    selected = arm.run(Fitness(), TEMPLATE, ledger, writer)
    assert ledger.executed_forecasts <= 30
    assert ledger.evaluator_calls >= 2
    assert selected["role_definition"] == "better"
    assert selected == json.loads((writer.dir / "selected_genotype.json").read_text())["genotype"]
    assert any(h["operation"] == "gepa_native_eval" for h in arm.history)
    assert any(h["full_opt_panel"] and h["genotype"] == {
        k: selected[k] for k in ("role_definition", "trait_formulations", "facet_formulations",
                              "critic_formulations")} for h in arm.history if h["parse_ok"])


def test_native_gepa_bounds_invalid_proposals_without_spending_forecasts(tmp_path):
    pytest.importorskip("gepa")
    from src.optim.arms.gepa_official import OfficialGEPA

    class Fitness:
        n_opt = 2
        tgt = (1,)
        forward_passes_per_candidate = 2
        Y = np.ones((2, 1), dtype=int)
        last_prompt_tokens = 0

        def _beliefs(self, g, idx):
            P = np.zeros((len(idx), 1, 5))
            P[:, :, 0] = 1
            return P, np.ones((len(idx), 1))

    class InvalidReflection:
        def complete(self, *args, **kwargs):
            return "```\nnot-json\n```", 2, 3

    ledger = BudgetLedger(5)
    arm = OfficialGEPA({"minibatch_respondents": 1, "max_invalid_proposals": 2},
                       np.random.default_rng(1), InvalidReflection(), TEMPLATE)
    arm.run(Fitness(), TEMPLATE, ledger, CellWriter(tmp_path, "invalid_gepa"))
    assert ledger.invalid_proposals <= 2
    assert ledger.executed_forecasts == 4  # seed plus two actual parent checks


def test_official_strict_genotype_rejects_template_repairs_and_zero_forecast_invalid(tmp_path):
    from src.optim.arms.gepa_official import RespondentAdapter, OfficialGEPA, strict_genotype
    from src.optim.genotype import GenotypeParseError

    candidate = {k: TEMPLATE[k] for k in ("role_definition", "trait_formulations",
                                           "facet_formulations", "critic_formulations")}
    candidate = json.loads(json.dumps(candidate))
    candidate["trait_formulations"][next(iter(candidate["trait_formulations"]))] = "  "
    with pytest.raises(GenotypeParseError):
        strict_genotype(json.dumps(candidate), TEMPLATE)
    candidate["trait_formulations"][next(iter(candidate["trait_formulations"]))] = 3
    with pytest.raises(GenotypeParseError):
        strict_genotype(json.dumps(candidate), TEMPLATE)
    candidate["trait_formulations"][next(iter(candidate["trait_formulations"]))] = "present"
    candidate["critic_formulations"] = ["wrong type"]
    with pytest.raises(GenotypeParseError):
        strict_genotype(json.dumps(candidate), TEMPLATE)
    candidate["critic_formulations"] = TEMPLATE["critic_formulations"]
    candidate["trait_formulations"].pop(next(iter(candidate["trait_formulations"])))
    with pytest.raises(GenotypeParseError):
        strict_genotype(json.dumps(candidate), TEMPLATE)

    class NoReadout:
        n_opt = 2
        tgt = (1,)
        forward_passes_per_candidate = 2

        def _beliefs(self, *args):
            raise AssertionError("invalid candidate reached readout")

    pytest.importorskip("gepa")
    arm = OfficialGEPA({}, np.random.default_rng(1), BadMutator(), TEMPLATE)
    arm.fitness, arm.ledger, arm.writer = NoReadout(), BudgetLedger(2, 2), CellWriter(tmp_path, "strict")
    batch = RespondentAdapter(arm).evaluate([0], {"persona_json": json.dumps(candidate)})
    assert batch.scores == [-1e6]
    assert arm.ledger.executed_forecasts == 0
    assert arm.ledger.invalid_proposals == 1


def test_strict_campaign_local_arm_rejects_partial_json_without_forecast(tmp_path):
    class PartialMutator:
        def complete(self, *args, **kwargs):
            g = {k: TEMPLATE[k] for k in ("role_definition", "trait_formulations",
                                          "facet_formulations", "critic_formulations")}
            g = json.loads(json.dumps(g))
            g["facet_formulations"].pop(next(iter(g["facet_formulations"])))
            return json.dumps(g), 4, 5

    ledger = BudgetLedger(2)
    arm = ARMS["random"]({"strict_genotype": True, "max_invalid_proposals": 1},
                          np.random.default_rng(1), PartialMutator(), TEMPLATE)
    arm.run(TinyFitness(), TEMPLATE, ledger, CellWriter(tmp_path, "strict_random"))
    assert ledger.executed_forecasts == TinyFitness.forward_passes_per_candidate
    assert ledger.invalid_proposals == 1
    assert arm.history[-1]["parse_ok"] is False
