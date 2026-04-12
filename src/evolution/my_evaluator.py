from external.evoprompt.evaluator import Evaluator  # Base из EvoPrompt

from src.utils.personality_match import (
    aggregate_stage_metrics,
    evaluate_participants_batch,
    normalize_participant_score,
)

from .utils import (
    genotype_to_evoprompt_str,
    parse_str_to_genotype,
    validate_and_repair_genotype,
)


class MyEvaluator(Evaluator):
    def __init__(self, args, task, model, fixed_modifiers, template_genotype=None, config=None):
        super().__init__(args)
        self.task = task
        self.model = model
        self.fixed_modifiers = fixed_modifiers
        self.dev_participants = []
        self.config = config
        self.template_genotype = template_genotype or {}
        cfg = config or {}
        self.participant_batch_size = int((cfg.get("evolution") or {}).get("participant_batch_size", 1) or 1)

    def forward(self, prompt_str, config=None):
        """
        Возвращает скалярный фитнес для эволюции (mean_similarity),
        но сохраняет полный stage-summary в self.last_detailed_scores.
        """
        if self.dev_participants is None:
            raise ValueError(
                "dev_participants не установлен! Установите evaluator.dev_participants перед вызовом forward()."
            )
        if hasattr(self.dev_participants, "empty"):
            if self.dev_participants.empty:
                raise ValueError("dev_participants пуст! Установите evaluator.dev_participants перед вызовом forward().")
        elif len(self.dev_participants) == 0:
            raise ValueError("dev_participants пуст! Установите evaluator.dev_participants перед вызовом forward().")

        if config is None:
            config = getattr(self, "config", None)
        if config is None:
            config = {"evolution": {"genotype_params": {}}}

        repaired_str = validate_and_repair_genotype(
            prompt_str,
            self.fixed_modifiers,
            self.template_genotype,
            config,
        )
        genotype = parse_str_to_genotype(
            repaired_str,
            self.fixed_modifiers,
            config,
            template_genotype=self.template_genotype,
        )
        canonical_prompt = genotype_to_evoprompt_str(genotype, config)

        raw_scores = evaluate_participants_batch(
            self.dev_participants,
            genotype,
            self.task,
            self.model,
            self.participant_batch_size,
        )
        participant_scores = [normalize_participant_score(score) for score in raw_scores]
        selected_facets = list((genotype.get("facet_formulations") or {}).keys())
        stage_metrics = aggregate_stage_metrics(participant_scores, selected_facets)

        self.last_detailed_scores = {
            "prompt": canonical_prompt,
            "stage_metrics": stage_metrics,
            "evaluation_scope": "generation",
            "dataset_split": "train",
            "metric_scope": "generation",
            "participants_total": len(participant_scores),
        }
        score = stage_metrics.get("summary", {}).get("mean_similarity")
        if isinstance(score, (int, float)):
            return float(score)
        # Для GA отсутствующие метрики должны быть явно худшими, но не "маскироваться нулём".
        return -1.0
