import json
import time
from pathlib import Path

import pandas as pd

from src.evolution.evoluter import GAEvoluter
from src.evolution.init_population import init_population
from src.evolution.my_evaluator import MyEvaluator
from src.evolution.parse_args import parse_args_from_yaml
from src.evolution.utils import (
    clean_evoprompt_response,
    parse_str_to_genotype,
    validate_and_repair_genotype,
)
from src.models.registry import get_model
from src.utils.personality_match import (
    aggregate_stage_metrics,
    evaluate_participants_batch,
    get_trait_question_blocks,
    normalize_participant_score,
)
from src.utils.save_result import save_jsonl, save_log
from src.utils.time import TimeEstimator, format_time

ARTIFACT_SCHEMA_VERSION = "2.0.0"


def _format_metric(value, default_text: str = "n/a") -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return default_text


def _to_json_safe(value):
    if value is None:
        return None
    if isinstance(value, (int, float, str, bool)):
        if isinstance(value, float) and (pd.isna(value) or value != value):
            return None
        return value
    return str(value)


def _extract_case_ids(df: pd.DataFrame) -> list:
    rows = []
    for idx, row in df.iterrows():
        case_val = row.get("case", idx)
        if pd.isna(case_val):
            case_val = idx
        rows.append(_to_json_safe(case_val))
    return rows


def _save_dataset_split_artifacts(results_dir: Path, cluster: int, train_participants: pd.DataFrame, test_participants: pd.DataFrame) -> dict:
    cluster_dir = results_dir / f"cluster_{cluster}"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    train_case_ids = _extract_case_ids(train_participants)
    test_case_ids = _extract_case_ids(test_participants)

    split_payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "cluster_id": cluster,
        "splits": {
            "train": train_case_ids,
            "test": test_case_ids,
        },
        "counts": {
            "train": len(train_case_ids),
            "test": len(test_case_ids),
        },
    }
    save_log(split_payload, cluster_dir, "dataset_split_ids.json")

    pd.DataFrame({"case": train_case_ids}).to_csv(cluster_dir / "train_case_ids.csv", index=False)
    pd.DataFrame({"case": test_case_ids}).to_csv(cluster_dir / "test_case_ids.csv", index=False)

    return {
        "dataset_split_ids_json": str(Path(f"cluster_{cluster}") / "dataset_split_ids.json").replace("\\", "/"),
        "train_case_ids_csv": str(Path(f"cluster_{cluster}") / "train_case_ids.csv").replace("\\", "/"),
        "test_case_ids_csv": str(Path(f"cluster_{cluster}") / "test_case_ids.csv").replace("\\", "/"),
    }


def _build_participant_metrics_row(case_id, score: dict, normalized_score: dict) -> dict:
    return {
        "case": _to_json_safe(case_id),
        "response_status": normalized_score.get("response_status"),
        "parse_status": normalized_score.get("parse_status"),
        "is_unparsable": bool(normalized_score.get("is_unparsable")),
        "model_answers_count": normalized_score.get("model_answers_count"),
        "valid_answer_pairs_count": normalized_score.get("valid_answer_pairs_count"),
        "similarity": normalized_score.get("similarity"),
        "avg_diff": normalized_score.get("avg_diff"),
        "pearson_corr": normalized_score.get("pearson_corr"),
        "mae_35": normalized_score.get("mae_35"),
        "similarity_35": normalized_score.get("similarity_35"),
        "pearson_35": normalized_score.get("pearson_35"),
        "kappa_35": normalized_score.get("kappa_35"),
        "mean_similarity_facets": normalized_score.get("mean_similarity_facets"),
        "mean_similarity_traits": normalized_score.get("mean_similarity_traits"),
        "mae_per_dim": normalized_score.get("mae_per_dim"),
        "similarity_per_dim": normalized_score.get("similarity_per_dim"),
        "answer_block_similarity": normalized_score.get("answer_block_similarity"),
        "model_answers": score.get("model_answers"),
        "simulated_ocean": score.get("simulated_ocean"),
        "error_message": normalized_score.get("error_message"),
    }


def _get_project_root():
    """Определяет корневую директорию проекта относительно расположения файла или текущей рабочей директории."""
    file_path = Path(__file__).resolve()
    candidate = file_path.parent.parent.parent
    if (candidate / "src").exists():
        return candidate

    cwd = Path.cwd()
    current = cwd
    while current != current.parent:
        if (current / "src").exists():
            return current
        current = current.parent
    return cwd


def _load_traits(config):
    """Загружает traits из config['prompt']['traits_path'] или из встроенного src.prompt.traits."""
    prompt_cfg = config.get("prompt") or {}
    path = prompt_cfg.get("traits_path")
    if path:
        p = Path(path)
        if not p.is_absolute():
            project_root = _get_project_root()
            p = project_root / p
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    from src.prompt.traits import traits

    return traits


def _load_facets(config):
    """Загружает facets из config['prompt']['facets_path'] или из встроенного src.prompt.facets."""
    prompt_cfg = config.get("prompt") or {}
    path = prompt_cfg.get("facets_path")
    if path:
        p = Path(path)
        if not p.is_absolute():
            project_root = _get_project_root()
            p = project_root / p
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    from src.prompt.facets import facets

    return facets


def _load_system(config):
    """Загружает system из config['prompt']['system_path'] или из встроенного src.prompt.system."""
    prompt_cfg = config.get("prompt") or {}
    path = prompt_cfg.get("system_path")
    if path:
        p = Path(path)
        if not p.is_absolute():
            project_root = _get_project_root()
            p = project_root / p
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    from src.prompt.system import system

    return system


def _load_trait_target_values(config):
    """Целевые значения черт по кластерам (для модификатора по совпадению)."""
    prompt_cfg = config.get("prompt") or {}
    path = prompt_cfg.get("traits_path")
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = _get_project_root() / p
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw = data.get("trait_target_values", {})
            return {int(k): v for k, v in raw.items()}
    from src.prompt.traits import trait_target_values

    return trait_target_values


def _load_facet_target_values(config):
    """Целевые значения фасетов по кластерам (для модификатора по совпадению)."""
    prompt_cfg = config.get("prompt") or {}
    path = prompt_cfg.get("facets_path")
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = _get_project_root() / p
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw = data.get("facet_target_values", {})
            return {int(k): v for k, v in raw.items()}
    from src.prompt.facets import facet_target_values

    return facet_target_values


def _evaluate_participants_on_test(
    test_participants,
    genotype,
    task,
    model,
    participant_batch_size,
    results_dir,
    cluster,
    selected_facets,
    csv_filename,
    participant_metrics_filename,
):
    scores = evaluate_participants_batch(
        test_participants,
        genotype,
        task,
        model,
        participant_batch_size,
    )

    participant_scores = []
    rows_answers = []
    participant_metrics_rows = []

    for (index, participant), score in zip(list(test_participants.iterrows()), scores):
        normalized_score = normalize_participant_score(score)
        participant_scores.append(normalized_score)

        model_answers = score.get("model_answers") or {}
        case_id = participant.get("case", index)
        row = {"case": case_id}
        for i in range(1, 121):
            row[f"i{i}"] = model_answers.get(i)
        rows_answers.append(row)
        participant_metrics_rows.append(_build_participant_metrics_row(case_id, score, normalized_score))

    cluster_dir = results_dir / f"cluster_{cluster}"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    if rows_answers:
        answers_df = pd.DataFrame(rows_answers)
    else:
        answers_df = pd.DataFrame(columns=["case", *[f"i{i}" for i in range(1, 121)]])

    csv_path = cluster_dir / csv_filename
    answers_df.to_csv(csv_path, index=False)

    save_jsonl(participant_metrics_rows, cluster_dir, participant_metrics_filename)

    stage_metrics = aggregate_stage_metrics(participant_scores, selected_facets)
    return {
        "participant_scores": participant_scores,
        "stage_metrics": stage_metrics,
        "answers_csv": str(Path(f"cluster_{cluster}") / csv_filename).replace("\\", "/"),
        "participant_metrics_jsonl": str(Path(f"cluster_{cluster}") / participant_metrics_filename).replace("\\", "/"),
    }


def _build_stage_payload(
    stage_metrics: dict,
    prompt,
    *,
    evaluation_scope: str,
    dataset_split: str,
    metric_scope: str,
    dataset_ids_artifact: str | None = None,
    answers_csv: str | None = None,
    participant_metrics_jsonl: str | None = None,
) -> dict:
    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "evaluation_scope": evaluation_scope,
        "dataset_split": dataset_split,
        "metric_scope": metric_scope,
        "dataset_ids_artifact": dataset_ids_artifact,
        "summary": stage_metrics.get("summary", {}),
        "trait_similarity": stage_metrics.get("trait_similarity", {}),
        "trait_similarity_valid_counts": stage_metrics.get("trait_similarity_valid_counts", {}),
        "facet_similarity": stage_metrics.get("facet_similarity", {}),
        "facet_similarity_valid_counts": stage_metrics.get("facet_similarity_valid_counts", {}),
        "answer_block_similarity": stage_metrics.get("answer_block_similarity", {}),
        "answer_block_valid_counts": stage_metrics.get("answer_block_valid_counts", {}),
        "selected_facets": stage_metrics.get("selected_facets", []),
        "trait_question_blocks": stage_metrics.get("trait_question_blocks", get_trait_question_blocks()),
        "prompt": prompt,
    }
    if answers_csv is not None:
        payload["answers_csv"] = answers_csv
    if participant_metrics_jsonl is not None:
        payload["participant_metrics_jsonl"] = participant_metrics_jsonl
    return payload


# ГЛАВНЫЙ ЦИКЛ ЭКСПЕРИМЕНТА
def run_experiment(config):
    """
    config: словарь с конфигурацией эксперимента, включающий:
        - data: настройки данных (file_path, cluster, num_participants)
        - model: настройки модели (name, provider, temperature)
        - evolution: настройки эволюционного алгоритма
        - experiment: настройки эксперимента (seed, save_every_generation)
        - results_dir: путь к директории для сохранения результатов
        - experiment_id: уникальный идентификатор эксперимента
    """
    results_dir = Path(config["results_dir"])
    experiment_id = config["experiment_id"]

    experiment_log = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "status": "started",
        "config": config,
        "clusters": {},
    }
    save_log(experiment_log, results_dir, "experiment_log.json")

    result_log = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "clusters": {},
    }
    save_log(result_log, results_dir, "result_log.json")

    traits = _load_traits(config)
    facets = _load_facets(config)
    system = _load_system(config)
    trait_target_values = _load_trait_target_values(config)
    facet_target_values = _load_facet_target_values(config)

    fixed_modifiers = system["intensity_modifiers"]

    print("📦 Загрузка модели...")
    model = get_model(config["model"])
    print(f"✅ Модель для симуляции загружена: {config['model'].get('model_name', 'неизвестно')}\n")

    evolution_model = None
    if "evolution" in config and config["evolution"].get("llm_for_evolution"):
        print("📦 Загрузка модели для эволюции...")
        evolution_model_config = {
            "model_name": config["evolution"]["llm_for_evolution"],
            "provider": config["evolution"].get("provider", "cloud"),
            "temperature": config["evolution"].get("temperature", 0.7),
            "timeout": config["evolution"].get("timeout")
            or config["model"].get("timeout")
            or config["model"].get("request_timeout"),
            "max_retries": config["evolution"].get("max_retries") or config["model"].get("max_retries"),
        }
        evolution_model = get_model(evolution_model_config)
        print(f"✅ Модель для эволюции загружена: {evolution_model_config['model_name']}\n")

    print("📂 Загрузка данных участников...")
    data_participants = pd.read_csv(config["data"]["file_path"])
    print(f"✅ Загружено участников: {len(data_participants)}\n")

    print("📋 Загрузка вопросов IPIP-NEO...")
    with open("data/IPIP-NEO/120/questions.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    ipip_neo_questions = data.get("questions")
    print(f"✅ Загружено вопросов: {len(ipip_neo_questions)}\n")

    task = {
        "task": system["task"],
        "ipip_neo": ipip_neo_questions,
        "response_format": system["response_format"],
    }

    clusters_list = config["data"]["clusters"]
    experiment_time_estimator = TimeEstimator(total_items=len(clusters_list))
    experiment_time_estimator.start()

    for cluster_idx, cluster in enumerate(clusters_list):
        cluster_start_time = time.time()
        experiment_time_estimator.start_item()

        print(f"\n{'#' * 70}")
        print(f"📊 ОБРАБОТКА КЛАСТЕРА: {cluster}")
        print(f"{'#' * 70}\n")

        cluster_trait_targets = trait_target_values.get(cluster, {})
        cluster_facet_targets = facet_target_values.get(cluster, {})
        trait_formulations = traits[cluster]
        facet_formulations = facets[cluster]

        base_genotype = {
            "role_definition": system["role"],
            "trait_formulations": trait_formulations,
            "facet_formulations": facet_formulations,
            "intensity_modifiers": system["intensity_modifiers"],
            "critic_formulations": system["critic_internal"],
            "template_structure": system["template_structure"],
            "trait_targets": {k: cluster_trait_targets[k] for k in trait_formulations if k in cluster_trait_targets},
            "facet_targets": {k: cluster_facet_targets[k] for k in facet_formulations if k in cluster_facet_targets},
        }
        genotype = base_genotype.copy()
        selected_facets = list(facet_formulations.keys())

        n_participants = config["data"]["num_participants"]
        total_participants = data_participants[data_participants["clusters"] == cluster].iloc[:n_participants]

        train_size = int(n_participants * 0.6)
        test_size = n_participants - train_size
        train_participants = total_participants.iloc[:train_size]
        test_participants = total_participants.iloc[train_size:]
        split_artifacts = _save_dataset_split_artifacts(results_dir, cluster, train_participants, test_participants)
        print(f"👥 Отобрано участников для кластера {cluster}: {len(total_participants)}")
        print(f"👥 Train: {train_size},  Test: {test_size}")

        participant_batch_size = int(
            (config.get("simulation") or {}).get("participant_batch_size")
            or (config.get("evolution") or {}).get("participant_batch_size", 1)
            or 1
        )

        print(f"\n📊 ПРОГОН БЕЗ ОПТИМИЗАЦИИ на Test (batch_size={participant_batch_size})")
        non_opt_results = _evaluate_participants_on_test(
            test_participants,
            base_genotype,
            task,
            model,
            participant_batch_size,
            results_dir,
            cluster,
            selected_facets,
            csv_filename="before_optimization_test_answers.csv",
            participant_metrics_filename="before_optimization_test_participants.jsonl",
        )
        before_stage = _build_stage_payload(
            stage_metrics=non_opt_results["stage_metrics"],
            prompt=base_genotype,
            evaluation_scope="before_optimization_test",
            dataset_split="test",
            metric_scope="stage",
            dataset_ids_artifact=split_artifacts.get("test_case_ids_csv"),
            answers_csv=non_opt_results["answers_csv"],
            participant_metrics_jsonl=non_opt_results["participant_metrics_jsonl"],
        )
        print("✅ Прогон без оптимизации завершён")
        print(f"  - Средняя схожесть: {_format_metric(before_stage['summary'].get('mean_similarity'))}")
        print(f"  - Средняя разница: {_format_metric(before_stage['summary'].get('mean_avg_diff'))}")
        print(f"  - Средняя корреляция Пирсона: {_format_metric(before_stage['summary'].get('mean_pearson_corr'))}\n")

        optimization_enabled = "evolution" in config and config["evolution"].get("algorithm")
        optimization_generations_stage = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "evaluation_scope": "generation",
            "dataset_split": "train",
            "metric_scope": "generation",
            "dataset_ids_artifact": split_artifacts.get("train_case_ids_csv"),
            "evolution_history_artifact": None,
            "generations": [],
        }
        if optimization_enabled:
            cluster_progress = experiment_time_estimator.get_progress_info(completed_items=cluster_idx)
            print(f"🧬 Запуск эволюционной оптимизации для кластера {cluster} | {cluster_progress}")

            evo_args = parse_args_from_yaml(config["evolution"])
            evaluator = MyEvaluator(
                evo_args,
                task,
                model,
                fixed_modifiers,
                template_genotype=base_genotype,
                config=config,
            )
            evaluator.dev_participants = train_participants
            evaluator.logger.info(
                f"MyEvaluator: установлено {len(train_participants)} участников для оценки (dev_participants)"
            )

            model_for_evolution = evolution_model if evolution_model is not None else model
            evoluter = GAEvoluter(evo_args, evaluator, evolution_model=model_for_evolution, config=config)
            init_population_records = init_population(base_genotype, config, evo_args.popsize, model_for_evolution)
            evoluter.set_initial_population(init_population_records)
            evoluter.evolute()

            best_str_raw = evoluter.population[0]
            best_str = clean_evoprompt_response(best_str_raw)
            best_str = validate_and_repair_genotype(best_str, fixed_modifiers, base_genotype, config)
            genotype = parse_str_to_genotype(best_str, fixed_modifiers, config, template_genotype=base_genotype)
            print("✅ Эволюция завершена. Лучший генотип сохранён.")

            cluster_dir = results_dir / f"cluster_{cluster}"
            cluster_dir.mkdir(parents=True, exist_ok=True)
            evolution_history_payload = {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "experiment_id": experiment_id,
                "cluster_id": cluster,
                "evaluation_scope": "generation",
                "dataset_split": "train",
                "metric_scope": "generation",
                "dataset_ids_artifact": split_artifacts.get("train_case_ids_csv"),
                "generations": getattr(evoluter, "generation_logs", []),
                "final_population": getattr(evoluter, "population_records", []),
            }
            save_log(evolution_history_payload, cluster_dir, "evolution_history.json")
            optimization_generations_stage["evolution_history_artifact"] = str(
                Path(f"cluster_{cluster}") / "evolution_history.json"
            ).replace("\\", "/")

            for gen_data in getattr(evoluter, "generation_logs", []):
                stage = gen_data.get("best_stage_summary") or {}
                optimization_generations_stage["generations"].append(
                    {
                        "generation": gen_data.get("generation"),
                        "evaluation_scope": gen_data.get("evaluation_scope", "generation"),
                        "dataset_split": gen_data.get("dataset_split", "train"),
                        "metric_scope": gen_data.get("metric_scope", "generation"),
                        "dataset_ids_artifact": split_artifacts.get("train_case_ids_csv"),
                        "population_size": gen_data.get("population_size"),
                        "valid_candidates_count": gen_data.get("valid_candidates_count"),
                        "unparsable_candidates_count": gen_data.get("unparsable_candidates_count"),
                        "best_candidate_id": gen_data.get("best_candidate_id"),
                        "best_score": gen_data.get("best_score"),
                        "mean_score": gen_data.get("mean_score"),
                        "median_score": gen_data.get("median_score"),
                        "min_score": gen_data.get("min_score"),
                        "max_score": gen_data.get("max_score"),
                        "std_score": gen_data.get("std_score"),
                        "best_prompt": gen_data.get("best_prompt"),
                        "status_breakdown": gen_data.get("status_breakdown", {}),
                        "population_metric_summary": gen_data.get("population_metric_summary", {}),
                        "summary": stage.get("summary", {}),
                        "trait_similarity": stage.get("trait_similarity", {}),
                        "trait_similarity_valid_counts": stage.get("trait_similarity_valid_counts", {}),
                        "facet_similarity": stage.get("facet_similarity", {}),
                        "facet_similarity_valid_counts": stage.get("facet_similarity_valid_counts", {}),
                        "answer_block_similarity": stage.get("answer_block_similarity", {}),
                        "answer_block_valid_counts": stage.get("answer_block_valid_counts", {}),
                        "selected_facets": stage.get("selected_facets", selected_facets),
                        "trait_question_blocks": stage.get("trait_question_blocks", get_trait_question_blocks()),
                    }
                )
        else:
            print("⚠️  Эволюционная оптимизация не включена в конфиге. Используется базовый генотип.")

        after_stage = None
        if optimization_enabled:
            print(f"📊 ОЦЕНКА ОПТИМИЗИРОВАННОГО ГЕНОТИПА на Test (batch_size={participant_batch_size})")
            opt_results = _evaluate_participants_on_test(
                test_participants,
                genotype,
                task,
                model,
                participant_batch_size,
                results_dir,
                cluster,
                selected_facets,
                csv_filename="after_optimization_test_answers.csv",
                participant_metrics_filename="after_optimization_test_participants.jsonl",
            )
            after_stage = _build_stage_payload(
                stage_metrics=opt_results["stage_metrics"],
                prompt=genotype,
                evaluation_scope="after_optimization_test",
                dataset_split="test",
                metric_scope="stage",
                dataset_ids_artifact=split_artifacts.get("test_case_ids_csv"),
                answers_csv=opt_results["answers_csv"],
                participant_metrics_jsonl=opt_results["participant_metrics_jsonl"],
            )
        else:
            print("⏭️  Повторный этап after_optimization_test пропущен (режим без оптимизации).")

        final_stage = after_stage if after_stage is not None else before_stage

        cluster_total_time = time.time() - cluster_start_time
        experiment_time_estimator.finish_item()
        experiment_progress = experiment_time_estimator.get_progress_info(completed_items=cluster_idx + 1)

        print(f"\n{'=' * 70}")
        print(f"📈 ИТОГОВЫЕ СРЕДНИЕ ПОКАЗАТЕЛИ КЛАСТЕРА {cluster}")
        print(f"{'=' * 70}")
        print(f"- Средняя схожесть (similarity): {_format_metric(final_stage['summary'].get('mean_similarity'))}")
        print(f"- Средняя разница (avg_diff): {_format_metric(final_stage['summary'].get('mean_avg_diff'))}")
        print(f"- Средняя корреляция Пирсона (pearson_corr): {_format_metric(final_stage['summary'].get('mean_pearson_corr'))}")
        print("  Five-factor (OCEAN+30):")
        print(f"  - MAE (mean |real−sim|): {_format_metric(final_stage['summary'].get('mean_mae_35'))}")
        print(f"  - Similarity по 35: {_format_metric(final_stage['summary'].get('mean_similarity_35'))}")
        print(f"  - Similarity по 30 фасетам: {_format_metric(final_stage['summary'].get('mean_similarity_facets'))}")
        print(f"  - Similarity по 5 чертам: {_format_metric(final_stage['summary'].get('mean_similarity_traits'))}")
        print(f"  - Pearson по 35: {_format_metric(final_stage['summary'].get('mean_pearson_35'))}")
        print(f"{'=' * 70}")
        print("⏱️  Статистика времени кластера:")
        print(f"- Общее время: {format_time(cluster_total_time)}")
        print(f"- Всего обработано участников: {n_participants}")
        print(f"- Прогресс эксперимента: {experiment_progress}")
        print(f"{'=' * 70}")
        print(f"✅ Обработка кластера {cluster} завершена\n")

        cluster_log = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "cluster_id": cluster,
            "start_time": cluster_start_time,
            "end_time": time.time(),
            "total_time": cluster_total_time,
            "participants_total": len(total_participants),
            "participants_train": len(train_participants),
            "participants_test": len(test_participants),
            "dataset_artifacts": split_artifacts,
            "stages": {
                "before_optimization_test": before_stage,
                "after_optimization_test": after_stage,
                "optimization_generations": optimization_generations_stage,
            },
        }

        result_log["clusters"][str(cluster)] = cluster_log
        save_log(result_log, results_dir, "result_log.json")

        experiment_log["clusters"][str(cluster)] = {
            "status": "completed",
            "total_time": cluster_total_time,
        }
        save_log(experiment_log, results_dir, "experiment_log.json")

    experiment_log["status"] = "completed"
    save_log(experiment_log, results_dir, "experiment_log.json")
    return experiment_log
