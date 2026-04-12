import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import scipy.stats as sps
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics import cohen_kappa_score

from src.utils import five_factor
from src.utils.parse import parse_response
from src.utils.prompt import build_full_prompt

# Имена 35 измерений и подмножества
OCEAN_AND_FACET_ORDER = five_factor.OCEAN_AND_FACET_ORDER
TRAIT_NAMES = five_factor.TRAIT_NAMES
FACET_NAMES = five_factor.FACET_NAMES

# Для IPIP-120 вопросы идут циклом N, E, O, A, C.
_TRAIT_BY_QID_MOD = {
    1: "neuroticism",
    2: "extraversion",
    3: "openness",
    4: "agreeableness",
    0: "conscientiousness",
}
_TRAIT_BLOCK_KEY = {
    "openness": "openness_items_24",
    "conscientiousness": "conscientiousness_items_24",
    "extraversion": "extraversion_items_24",
    "agreeableness": "agreeableness_items_24",
    "neuroticism": "neuroticism_items_24",
}

RESPONSE_STATUS_SUCCESS = "success"
RESPONSE_STATUS_UNPARSABLE = "unparsable"
RESPONSE_STATUS_NO_RESPONSE = "no_response"
RESPONSE_STATUS_INVALID = "invalid_response"
RESPONSE_STATUS_PARTIAL = "partial_response"
RESPONSE_STATUS_UNKNOWN = "unknown"

KNOWN_RESPONSE_STATUSES = {
    RESPONSE_STATUS_SUCCESS,
    RESPONSE_STATUS_UNPARSABLE,
    RESPONSE_STATUS_NO_RESPONSE,
    RESPONSE_STATUS_INVALID,
    RESPONSE_STATUS_PARTIAL,
}


def build_trait_question_blocks(total_questions: int = 120) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {trait: [] for trait in TRAIT_NAMES}
    for q_id in range(1, total_questions + 1):
        trait = _TRAIT_BY_QID_MOD[q_id % 5]
        grouped[trait].append(q_id)
    return {
        _TRAIT_BLOCK_KEY[trait]: grouped[trait]
        for trait in TRAIT_NAMES
    }


TRAIT_QUESTION_BLOCKS = build_trait_question_blocks(total_questions=120)


def get_trait_question_blocks() -> dict[str, list[int]]:
    return {k: list(v) for k, v in TRAIT_QUESTION_BLOCKS.items()}


def _is_valid_number(v) -> bool:
    return isinstance(v, (int, float)) and not (v != v or np.isnan(v))  # noqa: E711


def _safe_mean(arr: list[float]) -> float | None:
    if not arr:
        return None
    x = float(np.nanmean(arr))
    return None if (x != x or np.isnan(x)) else x  # noqa: E711


def _to_optional_float(v) -> float | None:
    return float(v) if _is_valid_number(v) else None


def _status_to_parse_status(response_status: str) -> str:
    if response_status == RESPONSE_STATUS_SUCCESS:
        return "parsed"
    if response_status == RESPONSE_STATUS_NO_RESPONSE:
        return "request_error"
    if response_status == RESPONSE_STATUS_INVALID:
        return "parse_error"
    if response_status == RESPONSE_STATUS_PARTIAL:
        return RESPONSE_STATUS_PARTIAL
    if response_status == RESPONSE_STATUS_UNPARSABLE:
        return RESPONSE_STATUS_UNPARSABLE
    return response_status or RESPONSE_STATUS_UNKNOWN


def _normalize_response_status(
    response_status: str | None,
    parse_status: str | None,
    model_answers_count: int,
    is_unparsable: bool,
) -> str:
    status = (response_status or "").strip().lower()
    if status == "parsed":
        status = RESPONSE_STATUS_SUCCESS
    elif status == "request_error":
        status = RESPONSE_STATUS_NO_RESPONSE
    elif status == "parse_error":
        status = RESPONSE_STATUS_INVALID

    if not status:
        parse = (parse_status or "").strip().lower()
        if parse == "parsed":
            status = RESPONSE_STATUS_SUCCESS
        elif parse == "request_error":
            status = RESPONSE_STATUS_NO_RESPONSE
        elif parse == "parse_error":
            status = RESPONSE_STATUS_INVALID
        elif parse == RESPONSE_STATUS_UNPARSABLE:
            status = RESPONSE_STATUS_UNPARSABLE
        elif parse == RESPONSE_STATUS_PARTIAL:
            status = RESPONSE_STATUS_PARTIAL

    if status not in KNOWN_RESPONSE_STATUSES:
        if is_unparsable:
            status = RESPONSE_STATUS_UNPARSABLE
        elif model_answers_count <= 0:
            status = RESPONSE_STATUS_NO_RESPONSE
        else:
            status = RESPONSE_STATUS_SUCCESS

    if status == RESPONSE_STATUS_SUCCESS and model_answers_count < 120:
        status = RESPONSE_STATUS_PARTIAL

    return status


def _value_to_category(x: float) -> str:
    """ low <33, average 33–66, high >66 """
    if x < 33:
        return "low"
    if x <= 66:
        return "average"
    return "high"


def compute_five_factor_metrics(
    real_flat: dict[str, float],
    simulated_flat: dict[str, float],
    keys: list[str] | None = None,
) -> dict[str, float | dict[str, float] | None]:
    """
    Сравнение реальных и смоделированных OCEAN+30.

    При отсутствии валидных данных возвращаются None-поля, чтобы
    не смешивать "нет данных" и "нулевое качество".
    """
    if keys is None:
        keys = [k for k in OCEAN_AND_FACET_ORDER if k in real_flat and k in simulated_flat]
    if not keys:
        return {
            "mae_35": None,
            "mae_per_dim": None,
            "similarity_35": None,
            "pearson_35": None,
            "kappa_35": None,
            "mean_similarity_facets": None,
            "mean_similarity_traits": None,
            "similarity_per_dim": None,
        }
    r_vec = np.array([real_flat[k] for k in keys])
    s_vec = np.array([simulated_flat[k] for k in keys])
    # MAE по каждому признаку и среднее
    mae_per_dim = {k: float(abs(real_flat[k] - simulated_flat[k])) for k in keys}
    mae_35 = float(np.mean(list(mae_per_dim.values())))
    # similarity по каждому измерению: 1 - |r-s|/100, затем среднее
    sim_per_dim = {k: 1.0 - abs(real_flat[k] - simulated_flat[k]) / 100.0 for k in keys}
    similarity_35 = float(np.mean(list(sim_per_dim.values())))

    # Pearson по 35 (или по keys)
    pearson_35 = None
    try:
        p = sps.pearsonr(r_vec, s_vec)[0]
        if _is_valid_number(p):
            pearson_35 = float(p)
    except Exception:
        pearson_35 = None

    # Cohen's kappa: категории low / average / high
    real_cat = [_value_to_category(real_flat[k]) for k in keys]
    sim_cat = [_value_to_category(simulated_flat[k]) for k in keys]
    kappa_35 = None
    try:
        k = cohen_kappa_score(real_cat, sim_cat)
        if _is_valid_number(k):
            kappa_35 = float(k)
    except Exception:
        kappa_35 = None

    # средняя similarity по 30 фасетам и по 5 чертам
    k_facets = [k for k in keys if k in FACET_NAMES]
    k_traits = [k for k in keys if k in TRAIT_NAMES]
    mean_similarity_facets = float(np.mean([sim_per_dim[k] for k in k_facets])) if k_facets else None
    mean_similarity_traits = float(np.mean([sim_per_dim[k] for k in k_traits])) if k_traits else None
    return {
        "mae_35": mae_35,
        "mae_per_dim": mae_per_dim,
        "similarity_35": similarity_35,
        "pearson_35": pearson_35,
        "kappa_35": kappa_35,
        "mean_similarity_facets": mean_similarity_facets,
        "mean_similarity_traits": mean_similarity_traits,
        "similarity_per_dim": sim_per_dim,
    }


def _empty_answer_block_similarity() -> dict[str, float | None]:
    return {k: None for k in TRAIT_QUESTION_BLOCKS}


def _build_unparsable_fitness(
    parse_status: str = RESPONSE_STATUS_UNPARSABLE,
    response_status: str | None = None,
    error_message: str | None = None,
) -> dict:
    status = response_status or parse_status
    payload = {
        "similarity": None,
        "avg_diff": None,
        "pearson_corr": None,
        "model_answers": None,
        "model_answers_count": 0,
        "valid_answer_pairs_count": 0,
        "simulated_ocean": None,
        "mae_35": None,
        "mae_per_dim": None,
        "similarity_35": None,
        "pearson_35": None,
        "kappa_35": None,
        "mean_similarity_facets": None,
        "mean_similarity_traits": None,
        "similarity_per_dim": None,
        "answer_block_similarity": _empty_answer_block_similarity(),
        "is_unparsable": True,
        "response_status": status,
        "parse_status": _status_to_parse_status(status),
    }
    if error_message:
        payload["error_message"] = error_message
    return payload


def _extract_response_text(response) -> str:
    if response is None:
        return ""

    content = response.content if hasattr(response, "content") else response
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Некоторые провайдеры возвращают список content-блоков.
        chunks = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(content)


def compute_answer_block_similarity(
    model_answers: dict[int, int] | None,
    participant,
) -> dict[str, float | None]:
    if not model_answers:
        return _empty_answer_block_similarity()

    out: dict[str, float | None] = {}
    for block_key, q_ids in TRAIT_QUESTION_BLOCKS.items():
        sims: list[float] = []
        for q_id in q_ids:
            model_ans = model_answers.get(q_id)
            if not isinstance(model_ans, int):
                continue
            human_ans = participant.get("i" + str(q_id))
            if human_ans is None or (isinstance(human_ans, float) and np.isnan(human_ans)):
                continue
            sims.append(1.0 - abs(model_ans - human_ans) / 4.0)
        out[block_key] = float(np.mean(sims)) if sims else None
    return out


def normalize_participant_score(score: dict) -> dict:
    pearson = score.get("pearson_corr")
    if isinstance(pearson, tuple):
        pearson = pearson[0]

    model_answers = score.get("model_answers")
    model_answers_count = score.get("model_answers_count")
    if not isinstance(model_answers_count, int):
        model_answers_count = len(model_answers) if isinstance(model_answers, dict) else 0

    is_unparsable = bool(score.get("is_unparsable", False))
    if "is_unparsable" not in score:
        is_unparsable = model_answers is None

    parse_status = score.get("parse_status")
    response_status = _normalize_response_status(
        response_status=score.get("response_status"),
        parse_status=parse_status,
        model_answers_count=model_answers_count,
        is_unparsable=is_unparsable,
    )

    if response_status in (RESPONSE_STATUS_NO_RESPONSE, RESPONSE_STATUS_UNPARSABLE, RESPONSE_STATUS_INVALID):
        is_unparsable = True

    parse_status_norm = _status_to_parse_status(response_status)

    valid_answer_pairs_count = score.get("valid_answer_pairs_count")
    if not isinstance(valid_answer_pairs_count, int):
        valid_answer_pairs_count = 0

    return {
        "similarity": _to_optional_float(score.get("similarity")),
        "avg_diff": _to_optional_float(score.get("avg_diff")),
        "pearson_corr": _to_optional_float(pearson),
        "mae_35": _to_optional_float(score.get("mae_35")),
        "mae_per_dim": score.get("mae_per_dim"),
        "similarity_35": _to_optional_float(score.get("similarity_35")),
        "pearson_35": _to_optional_float(score.get("pearson_35")),
        "kappa_35": _to_optional_float(score.get("kappa_35")),
        "mean_similarity_facets": _to_optional_float(score.get("mean_similarity_facets")),
        "mean_similarity_traits": _to_optional_float(score.get("mean_similarity_traits")),
        "similarity_per_dim": score.get("similarity_per_dim"),
        "answer_block_similarity": score.get("answer_block_similarity"),
        "is_unparsable": is_unparsable,
        "response_status": response_status,
        "parse_status": parse_status_norm,
        "model_answers_count": model_answers_count,
        "valid_answer_pairs_count": valid_answer_pairs_count,
        "error_message": score.get("error_message"),
    }


def aggregate_cluster_five_factor_metrics(participants_scores: list[dict]) -> dict[str, float | dict[str, float | None] | None]:
    """
    Усреднение five-factor метрик по тестовой выборке кластера.
    Учитываются только записи, где соответствующие поля валидны.
    """
    agg: dict[str, list[float]] = {
        "mae_35": [],
        "similarity_35": [],
        "pearson_35": [],
        "kappa_35": [],
        "mean_similarity_facets": [],
        "mean_similarity_traits": [],
    }
    for s in participants_scores:
        for k in agg:
            v = s.get(k)
            if _is_valid_number(v):
                agg[k].append(float(v))

    out: dict[str, float | dict[str, float | None] | None] = {
        f"mean_{k}": _safe_mean(agg[k]) for k in ("mae_35", "similarity_35", "pearson_35", "kappa_35")
    }
    out["mean_similarity_facets"] = _safe_mean(agg["mean_similarity_facets"])
    out["mean_similarity_traits"] = _safe_mean(agg["mean_similarity_traits"])

    # MAE по каждому из 35 признаков: усреднение по участникам
    mae_per_dim_collect: dict[str, list[float]] = {}
    for s in participants_scores:
        mpd = s.get("mae_per_dim")
        if isinstance(mpd, dict):
            for dim, val in mpd.items():
                if _is_valid_number(val):
                    mae_per_dim_collect.setdefault(dim, []).append(float(val))
    out["mean_mae_per_dim"] = {dim: _safe_mean(vals) for dim, vals in mae_per_dim_collect.items()}

    return out


def aggregate_stage_metrics(
    participants_scores: list[dict],
    selected_facets: list[str] | None = None,
) -> dict:
    """
    Сводные метрики по участникам.

    Важная семантика:
    - если метрику не на чем считать, возвращается None, а не 0.0;
    - средние считаются только по валидным значениям;
    - добавляются счётчики valid/excluded и статусы parseability.
    """
    selected_facets = list(selected_facets or [])
    participants_total = len(participants_scores)

    status_keys = [
        RESPONSE_STATUS_SUCCESS,
        RESPONSE_STATUS_UNPARSABLE,
        RESPONSE_STATUS_NO_RESPONSE,
        RESPONSE_STATUS_INVALID,
        RESPONSE_STATUS_PARTIAL,
        RESPONSE_STATUS_UNKNOWN,
    ]
    status_counts: dict[str, int] = {k: 0 for k in status_keys}
    for s in participants_scores:
        st = s.get("response_status")
        if st not in status_counts:
            st = RESPONSE_STATUS_UNKNOWN
        status_counts[st] += 1

    unparsable_count = sum(1 for s in participants_scores if bool(s.get("is_unparsable")))
    parsed_count = participants_total - unparsable_count

    def _metric_exclusion(metric_key: str, summary_key: str, aliases: list[str] | None = None) -> dict[str, int | float]:
        valid_count = sum(1 for s in participants_scores if _is_valid_number(s.get(metric_key)))
        excluded_count = participants_total - valid_count
        excluded_share = (excluded_count / participants_total) if participants_total else 0.0
        out = {
            f"{summary_key}_valid_count": valid_count,
            f"{summary_key}_excluded_count": excluded_count,
            f"{summary_key}_excluded_share": excluded_share,
        }
        for alias in aliases or []:
            out[f"{alias}_valid_count"] = valid_count
            out[f"{alias}_excluded_count"] = excluded_count
            out[f"{alias}_excluded_share"] = excluded_share
        return out

    def _metric_values(metric_key: str) -> list[float]:
        return [float(s.get(metric_key)) for s in participants_scores if _is_valid_number(s.get(metric_key))]

    summary: dict[str, int | float | None] = {
        "mean_similarity": _safe_mean(_metric_values("similarity")),
        "mean_avg_diff": _safe_mean(_metric_values("avg_diff")),
        "mean_pearson_corr": _safe_mean(_metric_values("pearson_corr")),
        "participants_total": participants_total,
        "participants_parsed_count": parsed_count,
        "participants_unparsable_count": unparsable_count,
        "participants_unparsable_share": (unparsable_count / participants_total) if participants_total else 0.0,
        "participants_success_count": status_counts[RESPONSE_STATUS_SUCCESS],
        "participants_partial_response_count": status_counts[RESPONSE_STATUS_PARTIAL],
        "participants_no_response_count": status_counts[RESPONSE_STATUS_NO_RESPONSE],
        "participants_invalid_response_count": status_counts[RESPONSE_STATUS_INVALID],
        "participants_unparsable_status_count": status_counts[RESPONSE_STATUS_UNPARSABLE],
        "participants_unknown_status_count": status_counts[RESPONSE_STATUS_UNKNOWN],
        "participants_with_model_answers_count": sum(
            1 for s in participants_scores if isinstance(s.get("model_answers_count"), int) and s.get("model_answers_count") > 0
        ),
        "participants_without_model_answers_count": sum(
            1 for s in participants_scores if not isinstance(s.get("model_answers_count"), int) or s.get("model_answers_count") <= 0
        ),
        "mean_model_answers_count": _safe_mean(
            [float(s.get("model_answers_count")) for s in participants_scores if isinstance(s.get("model_answers_count"), int)]
        ),
        "mean_valid_answer_pairs_count": _safe_mean(
            [float(s.get("valid_answer_pairs_count")) for s in participants_scores if isinstance(s.get("valid_answer_pairs_count"), int)]
        ),
    }

    for metric_key in (
        "similarity",
        "avg_diff",
        "pearson_corr",
        "mae_35",
        "similarity_35",
        "pearson_35",
        "kappa_35",
    ):
        summary.update(_metric_exclusion(metric_key, metric_key))
    summary.update(_metric_exclusion("mean_similarity_facets", "mean_similarity_facets", aliases=["similarity_facets"]))
    summary.update(_metric_exclusion("mean_similarity_traits", "mean_similarity_traits", aliases=["similarity_traits"]))

    ff = aggregate_cluster_five_factor_metrics(participants_scores)
    summary.update(
        {
            "mean_mae_35": ff.get("mean_mae_35"),
            "mean_similarity_35": ff.get("mean_similarity_35"),
            "mean_pearson_35": ff.get("mean_pearson_35"),
            "mean_kappa_35": ff.get("mean_kappa_35"),
            "mean_similarity_facets": ff.get("mean_similarity_facets"),
            "mean_similarity_traits": ff.get("mean_similarity_traits"),
            "mean_mae_per_dim": ff.get("mean_mae_per_dim"),
        }
    )

    trait_similarity: dict[str, float | None] = {}
    trait_similarity_valid_counts: dict[str, int] = {}
    for trait in TRAIT_NAMES:
        vals = []
        for s in participants_scores:
            per_dim = s.get("similarity_per_dim")
            if isinstance(per_dim, dict) and _is_valid_number(per_dim.get(trait)):
                vals.append(float(per_dim[trait]))
        trait_similarity[trait] = _safe_mean(vals)
        trait_similarity_valid_counts[trait] = len(vals)

    facet_similarity: dict[str, float | None] = {}
    facet_similarity_valid_counts: dict[str, int] = {}
    for facet in selected_facets:
        vals = []
        for s in participants_scores:
            per_dim = s.get("similarity_per_dim")
            if isinstance(per_dim, dict) and _is_valid_number(per_dim.get(facet)):
                vals.append(float(per_dim[facet]))
        facet_similarity[facet] = _safe_mean(vals)
        facet_similarity_valid_counts[facet] = len(vals)

    answer_block_similarity: dict[str, float | None] = {}
    answer_block_valid_counts: dict[str, int] = {}
    for block_key in TRAIT_QUESTION_BLOCKS:
        vals = []
        for s in participants_scores:
            blocks = s.get("answer_block_similarity")
            if isinstance(blocks, dict) and _is_valid_number(blocks.get(block_key)):
                vals.append(float(blocks[block_key]))
        answer_block_similarity[block_key] = _safe_mean(vals)
        answer_block_valid_counts[block_key] = len(vals)

    return {
        "summary": summary,
        "trait_similarity": trait_similarity,
        "trait_similarity_valid_counts": trait_similarity_valid_counts,
        "facet_similarity": facet_similarity,
        "facet_similarity_valid_counts": facet_similarity_valid_counts,
        "answer_block_similarity": answer_block_similarity,
        "answer_block_valid_counts": answer_block_valid_counts,
        "selected_facets": selected_facets,
        "trait_question_blocks": get_trait_question_blocks(),
    }


def evaluate_participants_batch(participants_df, genotype, task, model, batch_size=1):
    """
    Оценивает участников через fitness_function: по одному (batch_size<=1) или
    пачками с параллельными запросами к модели (batch_size>1).

    Вход:
        participants_df: DataFrame с участниками (как от .iterrows()).
        genotype, task, model: как для fitness_function.
        batch_size: 1 или None — последовательно; 5, 10, 20 — макс. число
                    одновременных запросов к модели (ThreadPoolExecutor).

    Выход:
        Список словарей score (как от fitness_function) в том же порядке, что
        participants_df.iterrows(). Средние считаются по ним так же, как раньше.
    """
    bs = int(batch_size or 0)
    if bs <= 1:
        return [
            fitness_function(participant, genotype, task, model)
            for _, participant in participants_df.iterrows()
        ]

    items = [(i, p) for i, p in participants_df.iterrows()]
    n = len(items)

    def _run_one(idx_p):
        _idx, p = idx_p
        return fitness_function(p, genotype, task, model)

    t0 = time.perf_counter()
    results = [None] * n
    done = 0
    progress_step = max(1, bs // 3)
    with ThreadPoolExecutor(max_workers=bs) as ex:
        futures = {ex.submit(_run_one, item): pos for pos, item in enumerate(items)}
        for fut in as_completed(futures):
            pos = futures[fut]
            try:
                results[pos] = fut.result()
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                print(f"[warn] Ошибка при обработке участника: {err}")
                results[pos] = _build_unparsable_fitness(
                    parse_status=RESPONSE_STATUS_NO_RESPONSE,
                    response_status=RESPONSE_STATUS_NO_RESPONSE,
                    error_message=err,
                )
            done += 1
            if done % progress_step == 0 or done == n:
                elapsed = time.perf_counter() - t0
                print(f"  [batch] completed {done}/{n}, max_concurrent={bs}, elapsed={elapsed:.1f}s")

    wall_s = time.perf_counter() - t0
    # Запросы уходят параллельно: до bs потоков вызывают model.generate одновременно
    print(f"  [batch] {n} participants, max_concurrent={bs}, wall_time={wall_s:.1f}s")
    return results


def fitness_function(participant, genotype, task, model):
    """
    Вычисляет соответствие модели реальному участнику по метрикам схожести ответов.

    Важное поведение:
    - "нет данных" возвращается как None;
    - "нулевая метрика" не подменяет недоступность;
    - response_status фиксирует причину (success/unparsable/no_response/invalid_response/partial_response).
    """
    prompt = build_full_prompt(genotype, task, participant)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt["system"]),
        ("human", prompt["human"]),
    ])
    try:
        response = model.generate(prompt_template)
        response_text = _extract_response_text(response)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        print(f"[warn] Ошибка запроса к модели: {err}")
        return _build_unparsable_fitness(
            parse_status=RESPONSE_STATUS_NO_RESPONSE,
            response_status=RESPONSE_STATUS_NO_RESPONSE,
            error_message=err,
        )

    if not isinstance(response_text, str) or not response_text.strip():
        return _build_unparsable_fitness(
            parse_status=RESPONSE_STATUS_NO_RESPONSE,
            response_status=RESPONSE_STATUS_NO_RESPONSE,
            error_message="Empty model response",
        )

    try:
        model_answers = parse_response(response_text)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        print(f"[warn] Ошибка парсинга ответа модели: {err}")
        return _build_unparsable_fitness(
            parse_status=RESPONSE_STATUS_INVALID,
            response_status=RESPONSE_STATUS_INVALID,
            error_message=err,
        )

    if model_answers is None:
        return _build_unparsable_fitness(
            parse_status=RESPONSE_STATUS_UNPARSABLE,
            response_status=RESPONSE_STATUS_UNPARSABLE,
        )

    if not isinstance(model_answers, dict) or not model_answers:
        return _build_unparsable_fitness(
            parse_status=RESPONSE_STATUS_INVALID,
            response_status=RESPONSE_STATUS_INVALID,
            error_message="Parsed answer is empty or invalid",
        )

    response_status = RESPONSE_STATUS_SUCCESS if len(model_answers) >= 120 else RESPONSE_STATUS_PARTIAL

    fitness = {
        "similarity": None,
        "avg_diff": None,
        "pearson_corr": None,
        "model_answers": model_answers,
        "model_answers_count": len(model_answers),
        "valid_answer_pairs_count": 0,
        "simulated_ocean": None,
        "mae_35": None,
        "mae_per_dim": None,
        "similarity_35": None,
        "pearson_35": None,
        "kappa_35": None,
        "mean_similarity_facets": None,
        "mean_similarity_traits": None,
        "similarity_per_dim": None,
        "answer_block_similarity": compute_answer_block_similarity(model_answers, participant),
        "is_unparsable": False,
        "response_status": response_status,
        "parse_status": _status_to_parse_status(response_status),
    }

    list_model_ans = []
    list_human_ans = []
    similarity_sum = 0.0
    avg_diff_sum = 0.0
    valid_count = 0
    for q_id, model_ans in model_answers.items():
        human_ans = participant.get("i" + str(q_id))
        if human_ans is None or (isinstance(human_ans, float) and np.isnan(human_ans)):
            continue
        list_model_ans.append(model_ans)
        list_human_ans.append(human_ans)
        similarity_sum += 1.0 - abs(model_ans - human_ans) / 4.0
        avg_diff_sum += abs(model_ans - human_ans)
        valid_count += 1

    fitness["valid_answer_pairs_count"] = valid_count
    if valid_count > 0:
        fitness["similarity"] = similarity_sum / valid_count
        fitness["avg_diff"] = avg_diff_sum / valid_count

    if len(list_model_ans) >= 2 and len(list_human_ans) >= 2:
        try:
            p = sps.pearsonr(list_model_ans, list_human_ans)[0]
            if _is_valid_number(p):
                fitness["pearson_corr"] = float(p)
        except Exception:
            fitness["pearson_corr"] = None

    simulated_ocean = five_factor.compute_ocean_facets(
        model_answers,
        participant.get("sex"),
        participant.get("age", 30),
        question=120,
    )
    fitness["simulated_ocean"] = simulated_ocean

    if simulated_ocean is not None and len(model_answers) >= 120:
        real_flat = {}
        for k in OCEAN_AND_FACET_ORDER:
            if k not in participant.index:
                continue
            v = participant[k]
            if v is None or (isinstance(v, float) and (np.isnan(v) or v != v)):
                continue
            try:
                real_flat[k] = float(v)
            except (TypeError, ValueError):
                pass

        common = [k for k in OCEAN_AND_FACET_ORDER if k in real_flat and k in simulated_ocean]
        if len(common) >= 30:
            m = compute_five_factor_metrics(real_flat, simulated_ocean, keys=common)
            fitness["mae_35"] = m["mae_35"]
            fitness["mae_per_dim"] = m["mae_per_dim"]
            fitness["similarity_35"] = m["similarity_35"]
            fitness["pearson_35"] = m["pearson_35"]
            fitness["kappa_35"] = m["kappa_35"]
            fitness["mean_similarity_facets"] = m["mean_similarity_facets"]
            fitness["mean_similarity_traits"] = m["mean_similarity_traits"]
            fitness["similarity_per_dim"] = m["similarity_per_dim"]

    return fitness
