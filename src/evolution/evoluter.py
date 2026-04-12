import random
import statistics

import numpy as np

from src.evolution.operators import my_crossover, my_mutate
from src.evolution.utils import (
    clean_evoprompt_response,
    genotype_to_evoprompt_str,
    parse_str_to_genotype,
    validate_and_repair_genotype,
)
from src.utils.time import TimeEstimator, format_time


def _is_valid_number(v) -> bool:
    return isinstance(v, (int, float)) and np.isfinite(v)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _safe_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if len(values) == 1 else None
    return float(np.std(values))


def _format_score(v: float | None) -> str:
    return f"{v:.4f}" if _is_valid_number(v) else "n/a"


class Evoluter:
    """
    Базовый класс для эволюции (адаптировано из EvoPrompt).
    """

    def __init__(self, args, evaluator, evolution_model=None, config=None):
        self.args = args
        self.evaluator = evaluator
        self.evolution_model = evolution_model
        self.config = config
        self.population = []
        self.scores: list[float | None] = []
        self.generation_logs = []

    def evolute(self):
        raise NotImplementedError("Реализуется в подклассах")


class GAEvoluter(Evoluter):
    """
    Genetic Algorithm эволюция (адаптировано под нашу задачу).
    """

    def __init__(self, args, evaluator, evolution_model=None, config=None):
        super().__init__(args, evaluator, evolution_model, config)
        self.population_size = args.popsize
        self.num_generations = args.budget
        self.mutation_prob = args.mutation_prob
        self.crossover_prob = args.crossover_prob
        self.selection_method = args.sel_mode.lower()
        self.detailed_scores_per_prompt = []
        self.population_records: list[dict] = []
        self._candidate_seq = 0
        self._operation_seq = {"mutation": 0, "crossover": 0}
        self.best_tie_break_policy = "keep_earlier_candidate_on_equal_score"
        self.best_so_far_prompt: str | None = None
        self.best_so_far_score: float | None = None
        self.best_so_far_generation: int | None = None
        self.best_so_far_candidate_id: str | None = None
        self.best_so_far_parent_ids: list[str] = []
        self.best_so_far_operation: str | None = None
        self.best_so_far_summary: dict = {}
        self.final_population_best_prompt: str | None = None
        self.final_population_best_score: float | None = None
        self.final_population_best_candidate_id: str | None = None

    def _reset_best_tracking(self):
        self.best_so_far_prompt = None
        self.best_so_far_score = None
        self.best_so_far_generation = None
        self.best_so_far_candidate_id = None
        self.best_so_far_parent_ids = []
        self.best_so_far_operation = None
        self.best_so_far_summary = {}
        self.final_population_best_prompt = None
        self.final_population_best_score = None
        self.final_population_best_candidate_id = None

    def _next_candidate_id(self, generation: int) -> str:
        cid = f"g{generation}_c{self._candidate_seq:04d}"
        self._candidate_seq += 1
        return cid

    def _next_operation_id(self, kind: str, generation: int) -> str:
        self._operation_seq[kind] = self._operation_seq.get(kind, 0) + 1
        return f"{kind}_g{generation}_{self._operation_seq[kind]:04d}"

    def set_initial_population(self, population: list[str]):
        self.population = list(population)
        self.population_records = []
        self._reset_best_tracking()
        for prompt in self.population:
            self.population_records.append(
                {
                    "candidate_id": self._next_candidate_id(0),
                    "generation": 0,
                    "prompt": prompt,
                    "parent_ids": [],
                    "operation": "init",
                    "mutation_prompt_id": None,
                    "mutation_context": None,
                    "source_prompt_id": None,
                }
            )

    def _ensure_population_records(self, generation: int):
        if self.population_records and len(self.population_records) == len(self.population):
            return
        self.population_records = []
        for prompt in self.population:
            self.population_records.append(
                {
                    "candidate_id": self._next_candidate_id(generation),
                    "generation": generation,
                    "prompt": prompt,
                    "parent_ids": [],
                    "operation": "unknown",
                    "mutation_prompt_id": None,
                    "mutation_context": None,
                    "source_prompt_id": None,
                }
            )

    def _should_update_best_so_far(self, score: float | None) -> bool:
        if not _is_valid_number(score):
            return False
        if not _is_valid_number(self.best_so_far_score):
            return True
        # Tie-break rule: keep the earlier found best for deterministic behavior.
        return float(score) > float(self.best_so_far_score)

    def _update_best_so_far(self, generation: int):
        if not self.population_records or not self.scores:
            return
        best_record = self.population_records[0]
        best_score = self.scores[0]
        if not self._should_update_best_so_far(best_score):
            return

        detailed = self.detailed_scores_per_prompt[0] if self.detailed_scores_per_prompt else {}
        stage_metrics = detailed.get("stage_metrics") if isinstance(detailed, dict) else {}
        if not isinstance(stage_metrics, dict):
            stage_metrics = {}
        summary = stage_metrics.get("summary") if isinstance(stage_metrics, dict) else {}
        if not isinstance(summary, dict):
            summary = {}

        self.best_so_far_prompt = best_record.get("prompt", "")
        self.best_so_far_score = float(best_score) if _is_valid_number(best_score) else None
        self.best_so_far_generation = generation
        self.best_so_far_candidate_id = best_record.get("candidate_id")
        self.best_so_far_parent_ids = list(best_record.get("parent_ids") or [])
        self.best_so_far_operation = best_record.get("operation")
        self.best_so_far_summary = dict(summary)

    def _capture_final_population_best(self):
        if not self.population_records:
            self.final_population_best_prompt = None
            self.final_population_best_score = None
            self.final_population_best_candidate_id = None
            return
        self.final_population_best_prompt = self.population_records[0].get("prompt", "")
        self.final_population_best_score = self.scores[0] if self.scores else None
        self.final_population_best_candidate_id = self.population_records[0].get("candidate_id")

    def evaluate_population(self, generation: int):
        """
        Оценивает всю популяцию и сохраняет детальные stage-метрики
        для каждого кандидата.
        """
        self._ensure_population_records(generation)

        self.scores = []
        self.detailed_scores_per_prompt = []

        for record in self.population_records:
            prompt_str = record.get("prompt", "")
            raw_score = self.evaluator.forward(prompt_str, self.config)
            score = float(raw_score) if _is_valid_number(raw_score) else None
            self.scores.append(score)

            if hasattr(self.evaluator, "last_detailed_scores"):
                self.detailed_scores_per_prompt.append(self.evaluator.last_detailed_scores.copy())
            else:
                self.detailed_scores_per_prompt.append(
                    {
                        "prompt": prompt_str,
                        "stage_metrics": {
                            "summary": {
                                "mean_similarity": score,
                                "mean_avg_diff": None,
                                "mean_pearson_corr": None,
                                "mean_mae_35": None,
                                "mean_similarity_35": None,
                                "mean_pearson_35": None,
                                "mean_kappa_35": None,
                                "mean_similarity_facets": None,
                                "mean_similarity_traits": None,
                            },
                            "trait_similarity": {},
                            "facet_similarity": {},
                            "answer_block_similarity": {},
                            "selected_facets": [],
                            "trait_question_blocks": {},
                        },
                    }
                )

        def _score_sort_value(idx: int) -> float:
            v = self.scores[idx]
            return float(v) if _is_valid_number(v) else float("-inf")

        sorted_idx = sorted(range(len(self.population_records)), key=_score_sort_value, reverse=True)
        self.population_records = [self.population_records[i] for i in sorted_idx]
        self.scores = [self.scores[i] for i in sorted_idx]
        self.detailed_scores_per_prompt = [self.detailed_scores_per_prompt[i] for i in sorted_idx]
        self.population = [r.get("prompt", "") for r in self.population_records]

    def _infer_candidate_status(self, summary: dict, score: float | None) -> str:
        success_count = int(summary.get("participants_success_count") or 0)
        partial_count = int(summary.get("participants_partial_response_count") or 0)
        no_response_count = int(summary.get("participants_no_response_count") or 0)
        invalid_count = int(summary.get("participants_invalid_response_count") or 0)
        unparsable_count = int(summary.get("participants_unparsable_status_count") or summary.get("participants_unparsable_count") or 0)

        if success_count > 0 and partial_count == 0 and no_response_count == 0 and invalid_count == 0 and unparsable_count == 0:
            return "success"
        if partial_count > 0:
            return "partial_response"
        if no_response_count > 0:
            return "no_response"
        if invalid_count > 0:
            return "invalid_response"
        if unparsable_count > 0:
            return "unparsable"
        if _is_valid_number(score):
            return "success"
        return "unknown"

    def _build_candidate_log(
        self,
        generation: int,
        rank: int,
        record: dict,
        score: float | None,
        detailed: dict,
    ) -> dict:
        stage_metrics = detailed.get("stage_metrics") if isinstance(detailed, dict) else {}
        if not isinstance(stage_metrics, dict):
            stage_metrics = {}
        summary = stage_metrics.get("summary") if isinstance(stage_metrics, dict) else {}
        if not isinstance(summary, dict):
            summary = {}
        status = self._infer_candidate_status(summary, score)
        return {
            "candidate_id": record.get("candidate_id"),
            "generation": generation,
            "rank": rank,
            "parent_ids": list(record.get("parent_ids") or []),
            "operation": record.get("operation"),
            "mutation_prompt_id": record.get("mutation_prompt_id"),
            "mutation_context": record.get("mutation_context"),
            "source_prompt_id": record.get("source_prompt_id"),
            "evaluation_scope": "generation",
            "dataset_split": "train",
            "metric_scope": "generation",
            "status": status,
            "score": score,
            "prompt": record.get("prompt"),
            "summary": summary,
            "trait_similarity": stage_metrics.get("trait_similarity", {}),
            "facet_similarity": stage_metrics.get("facet_similarity", {}),
            "answer_block_similarity": stage_metrics.get("answer_block_similarity", {}),
            "selected_facets": stage_metrics.get("selected_facets", []),
            "trait_question_blocks": stage_metrics.get("trait_question_blocks", {}),
        }

    def _build_population_summary(self, candidates: list[dict]) -> dict:
        scores = [float(c["score"]) for c in candidates if _is_valid_number(c.get("score"))]
        status_breakdown: dict[str, int] = {}
        for c in candidates:
            status = c.get("status") or "unknown"
            status_breakdown[status] = status_breakdown.get(status, 0) + 1

        best_candidate = candidates[0] if candidates else {}
        best_score = best_candidate.get("score") if candidates else None
        best_candidate_id = best_candidate.get("candidate_id") if candidates else None

        metric_keys = [
            "mean_similarity",
            "mean_avg_diff",
            "mean_pearson_corr",
            "mean_mae_35",
            "mean_similarity_35",
            "mean_pearson_35",
            "mean_kappa_35",
            "mean_similarity_facets",
            "mean_similarity_traits",
        ]
        population_metric_summary: dict[str, float | None] = {}
        for mk in metric_keys:
            vals = []
            for c in candidates:
                summary = c.get("summary")
                if isinstance(summary, dict) and _is_valid_number(summary.get(mk)):
                    vals.append(float(summary[mk]))
            population_metric_summary[f"population_{mk}"] = _safe_mean(vals)

        return {
            "population_size": len(candidates),
            "valid_candidates_count": len(scores),
            "unparsable_candidates_count": sum(
                1 for c in candidates if c.get("status") in {"unparsable", "no_response", "invalid_response"}
            ),
            "mean_score": _safe_mean(scores),
            "median_score": float(statistics.median(scores)) if scores else None,
            "min_score": min(scores) if scores else None,
            "max_score": max(scores) if scores else None,
            "std_score": _safe_std(scores),
            "best_score": best_score if _is_valid_number(best_score) else None,
            "best_candidate_id": best_candidate_id,
            "status_breakdown": status_breakdown,
            "population_metric_summary": population_metric_summary,
        }

    def _append_generation_log(self, generation: int):
        candidates = []
        for i, record in enumerate(self.population_records):
            detailed = self.detailed_scores_per_prompt[i] if i < len(self.detailed_scores_per_prompt) else {}
            score = self.scores[i] if i < len(self.scores) else None
            candidates.append(self._build_candidate_log(generation, i + 1, record, score, detailed))

        pop_summary = self._build_population_summary(candidates)
        best_detailed = self.detailed_scores_per_prompt[0] if self.detailed_scores_per_prompt else {}
        best_of_generation_prompt = candidates[0].get("prompt") if candidates else ""
        best_of_generation_score = pop_summary.get("best_score")
        best_of_generation_candidate_id = pop_summary.get("best_candidate_id")

        self.generation_logs.append(
            {
                "generation": generation,
                "evaluation_scope": "generation",
                "dataset_split": "train",
                "metric_scope": "generation",
                "population_size": pop_summary.get("population_size"),
                "valid_candidates_count": pop_summary.get("valid_candidates_count"),
                "unparsable_candidates_count": pop_summary.get("unparsable_candidates_count"),
                "mean_score": pop_summary.get("mean_score"),
                "median_score": pop_summary.get("median_score"),
                "min_score": pop_summary.get("min_score"),
                "max_score": pop_summary.get("max_score"),
                "std_score": pop_summary.get("std_score"),
                "best_candidate_id": best_of_generation_candidate_id,
                "best_prompt": best_of_generation_prompt,
                "best_score": best_of_generation_score,
                "best_of_generation_candidate_id": best_of_generation_candidate_id,
                "best_of_generation_prompt": best_of_generation_prompt,
                "best_of_generation_score": best_of_generation_score,
                "best_so_far_candidate_id": self.best_so_far_candidate_id,
                "best_so_far_prompt": self.best_so_far_prompt,
                "best_so_far_score": self.best_so_far_score,
                "best_so_far_generation": self.best_so_far_generation,
                "best_so_far_parent_ids": list(self.best_so_far_parent_ids or []),
                "best_so_far_operation": self.best_so_far_operation,
                "best_so_far_summary": dict(self.best_so_far_summary or {}),
                "best_tie_break_policy": self.best_tie_break_policy,
                "best_stage_summary": best_detailed.get("stage_metrics") or {},
                "population_metric_summary": pop_summary.get("population_metric_summary", {}),
                "status_breakdown": pop_summary.get("status_breakdown", {}),
                "candidates": candidates,
            }
        )

    def _score_for_selection(self, idx: int) -> float:
        score = self.scores[idx]
        return float(score) if _is_valid_number(score) else -1.0

    def select_parent_indices(self) -> tuple[int, int]:
        n = len(self.population_records)
        if n <= 1:
            return 0, 0
        if n == 2:
            return 0, 1

        if self.selection_method == "tournament":
            tournament_size = min(3, n - 1)

            def tournament():
                candidates = random.sample(range(n), tournament_size)
                return max(candidates, key=self._score_for_selection)

            parent1_idx = tournament()
            parent2_idx = parent1_idx
            for _ in range(50):
                parent2_idx = tournament()
                if parent2_idx != parent1_idx:
                    break
            if parent2_idx == parent1_idx:
                parent2_idx = (parent1_idx + 1) % n
            return parent1_idx, parent2_idx

        if self.selection_method == "roulette":
            score_vals = [self._score_for_selection(i) for i in range(n)]
            min_score = min(score_vals)
            adjusted = [s - min_score + 0.001 for s in score_vals]
            total = sum(adjusted)
            if total <= 0:
                return tuple(random.sample(range(n), 2))
            probs = [s / total for s in adjusted]
            parent1_idx = random.choices(range(n), weights=probs, k=1)[0]
            parent2_idx = random.choices(range(n), weights=probs, k=1)[0]
            while parent1_idx == parent2_idx and n > 1:
                parent2_idx = random.choices(range(n), weights=probs, k=1)[0]
            return parent1_idx, parent2_idx

        return tuple(random.sample(range(n), 2))

    def _normalize_child_prompt(self, child_prompt: str, template_prompt: str) -> str:
        child_prompt = clean_evoprompt_response(child_prompt)
        if not self.config:
            return child_prompt
        try:
            template_genotype = parse_str_to_genotype(template_prompt, self.evaluator.fixed_modifiers, self.config)
            child_prompt = validate_and_repair_genotype(
                child_prompt,
                self.evaluator.fixed_modifiers,
                template_genotype,
                self.config,
            )
            repaired_genotype = parse_str_to_genotype(child_prompt, self.evaluator.fixed_modifiers, self.config)
            return genotype_to_evoprompt_str(repaired_genotype, self.config)
        except Exception:
            return template_prompt

    def evolute(self):
        print(f"🧬 Старт GA эволюции: pop_size={self.population_size}, generations={self.num_generations}")

        if not self.population_records:
            self.set_initial_population(self.population)

        time_estimator = TimeEstimator(total_items=self.num_generations)
        time_estimator.start()
        time_estimator.start_item()

        self.evaluate_population(generation=0)
        self._update_best_so_far(generation=0)
        time_estimator.finish_item()
        self._append_generation_log(generation=0)

        progress_info = time_estimator.get_progress_info(completed_items=1)
        gen0_best = self.scores[0] if self.scores else None
        gen0_mean = _safe_mean([float(s) for s in self.scores if _is_valid_number(s)])
        print(f"Gen 0: best = {_format_score(gen0_best)}, mean = {_format_score(gen0_mean)} | {progress_info}")

        for gen in range(1, self.num_generations):
            time_estimator.start_item()
            new_records: list[dict] = []

            elite_count = min(2, len(self.population_records))
            for elite_rank, elite in enumerate(self.population_records[:elite_count], start=1):
                new_records.append(
                    {
                        "candidate_id": self._next_candidate_id(gen),
                        "generation": gen,
                        "prompt": elite.get("prompt", ""),
                        "parent_ids": [elite.get("candidate_id")],
                        "operation": "elite",
                        "mutation_prompt_id": None,
                        "mutation_context": {"elite_rank": elite_rank},
                        "source_prompt_id": elite.get("candidate_id"),
                    }
                )

            while len(new_records) < self.population_size:
                p1_idx, p2_idx = self.select_parent_indices()
                parent1 = self.population_records[p1_idx]
                parent2 = self.population_records[p2_idx]
                parent1_prompt = parent1.get("prompt", "")
                parent2_prompt = parent2.get("prompt", "")

                crossover_used = random.random() < self.crossover_prob
                crossover_prompt_id = None

                if crossover_used:
                    crossover_prompt_id = self._next_operation_id("crossover", gen)
                    child1_raw, child2_raw = my_crossover(
                        parent1_prompt,
                        parent2_prompt,
                        self.evolution_model,
                        self.config,
                        self.evaluator.fixed_modifiers,
                    )
                    base_operation = "crossover"
                else:
                    child1_raw, child2_raw = parent1_prompt, parent2_prompt
                    base_operation = "clone"

                children = [
                    (child1_raw, parent1.get("candidate_id")),
                    (child2_raw, parent2.get("candidate_id")),
                ]

                for child_idx, (child_raw, source_parent_id) in enumerate(children, start=1):
                    if len(new_records) >= self.population_size:
                        break

                    mutation_in = child_raw
                    mutation_out = my_mutate(mutation_in, self.mutation_prob, self.evolution_model, self.config)
                    mutation_applied = mutation_out != mutation_in
                    mutation_prompt_id = self._next_operation_id("mutation", gen) if mutation_applied else None

                    if base_operation == "crossover":
                        operation = "crossover+mutation" if mutation_applied else "crossover"
                    else:
                        operation = "mutation" if mutation_applied else "clone"

                    template_prompt = self.population[0] if self.population else mutation_in
                    normalized_child = self._normalize_child_prompt(mutation_out, template_prompt)
                    if crossover_used:
                        parent_ids = [parent1.get("candidate_id"), parent2.get("candidate_id")]
                    else:
                        parent_ids = [source_parent_id]

                    new_records.append(
                        {
                            "candidate_id": self._next_candidate_id(gen),
                            "generation": gen,
                            "prompt": normalized_child,
                            "parent_ids": [pid for pid in parent_ids if pid is not None],
                            "operation": operation,
                            "mutation_prompt_id": mutation_prompt_id,
                            "mutation_context": {
                                "crossover_used": crossover_used,
                                "crossover_prompt_id": crossover_prompt_id,
                                "child_index": child_idx,
                                "mutation_prob": self.mutation_prob,
                            },
                            "source_prompt_id": source_parent_id,
                        }
                    )

            self.population_records = new_records[: self.population_size]
            self.population = [r.get("prompt", "") for r in self.population_records]

            self.evaluate_population(generation=gen)
            self._update_best_so_far(generation=gen)
            time_estimator.finish_item()
            self._append_generation_log(generation=gen)

            progress_info = time_estimator.get_progress_info(completed_items=gen + 1)
            gen_best = self.scores[0] if self.scores else None
            gen_mean = _safe_mean([float(s) for s in self.scores if _is_valid_number(s)])
            print(f"Gen {gen}: best = {_format_score(gen_best)}, mean = {_format_score(gen_mean)} | {progress_info}")

        self._capture_final_population_best()
        total_evolution_time = time_estimator.get_elapsed()
        final_best = self.final_population_best_score
        global_best = self.best_so_far_score
        global_best_generation = self.best_so_far_generation
        print(
            "🧬 Эволюция завершена. "
            f"best_last_generation={_format_score(final_best)}, "
            f"best_so_far={_format_score(global_best)} (generation={global_best_generation}) | "
            f"Общее время: {format_time(total_evolution_time)}"
        )
