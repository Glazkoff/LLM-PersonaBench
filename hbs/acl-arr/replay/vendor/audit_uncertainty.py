"""Independent replay of the registered R2 primary respondent uncertainty.

This module imports no experiment scorer. Its caller must first verify the
compare-score gate and independently reconstruct labels and forecast panels.
"""
from __future__ import annotations

import math
import numpy as np

FAMILIES = ("economic_preferences", "heuristics_cognitive_bias")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def same(a, b, label):
    if a is None or b is None:
        require(a is b, f"{label}: null mismatch")
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), f"{label}: interval shape")
        for x, y in zip(a, b):
            same(x, y, label)
    else:
        require(math.isfinite(float(a)) and math.isfinite(float(b))
                and math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-10),
                f"{label}: numerical mismatch")


def losses(probabilities, labels, kind):
    """One proper loss per observed person, in original respondent order."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    require(p.ndim == 2 and y.shape == (len(p),) and p.shape[1] >= 2,
            "probability/label axes")
    mask = y >= 0
    if not mask.any():
        return np.empty(0)
    q, observed = p[mask], y[mask]
    require(np.all((observed >= 0) & (observed < q.shape[1])), "label outside options")
    if kind == "ordinal":
        cumulative = np.cumsum(q, axis=1)[:, :-1]
        target = observed[:, None] <= np.arange(q.shape[1] - 1)
        return np.mean(np.square(cumulative - target), axis=1)
    with np.errstate(divide="ignore"):
        return -np.log(q[np.arange(len(observed)), observed])


def aggregate(values, denominators, families, rng, draws, min_n=30):
    """Shared respondent indices/signs; endpoint means, then equal family means."""
    require(set(values) == set(denominators) == set(families)
            and set(families.values()) == set(FAMILIES), "aggregate endpoint/family set")
    n = len(next(iter(values.values())))
    require(all(len(v) == n for v in values.values()) and n > 0, "respondent axis")
    require(all(math.isfinite(float(d)) and d > 1e-6 for d in denominators.values()),
            "frozen CAL denominator")
    observed = {key: int(np.count_nonzero(~np.isnan(value))) for key, value in values.items()}
    if any(count < min_n for count in observed.values()):
        return {"status": "unavailable_endpoint_below_min_n", "estimate": None,
                "paired_people_ci95_conditional_valid": None, "paired_signflip_p": None,
                "failed_bootstrap_fraction": None, "observed_test_n": observed}
    if any(not np.isfinite(value[~np.isnan(value)]).all() for value in values.values()):
        return {"status": "unavailable_nonfinite_primary_loss", "estimate": None,
                "paired_people_ci95_conditional_valid": None, "paired_signflip_p": None,
                "failed_bootstrap_fraction": None, "observed_test_n": observed}

    def statistic(indices, signs=None):
        family_values = []
        for family in FAMILIES:
            per_endpoint = []
            for endpoint, label in families.items():
                if label != family:
                    continue
                sample = values[endpoint][indices]
                keep = ~np.isnan(sample)
                if not keep.any():
                    return None
                if signs is not None:
                    sample = sample * signs[indices]
                per_endpoint.append(float(sample[keep].mean() / denominators[endpoint]))
            family_values.append(float(np.mean(per_endpoint)))
        return float(np.mean(family_values))

    point = statistic(np.arange(n))
    bootstrap, null = [], []
    failed = 0
    for _ in range(draws):
        indices = rng.integers(n, size=n)
        value = statistic(indices)
        if value is None:
            failed += 1
        else:
            bootstrap.append(value)
        signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)
        null.append(statistic(np.arange(n), signs))
    interval = np.quantile(bootstrap, [.025, .975]).tolist() if bootstrap else None
    p = float((1 + np.count_nonzero(np.abs(null) >= abs(point))) / (draws + 1))
    return {"status": "available" if bootstrap else "unavailable_all_bootstrap_draws_empty_endpoint",
            "estimate": point, "paired_people_ci95_conditional_valid": interval,
            "paired_signflip_p": p, "failed_bootstrap_fraction": failed / draws,
            "valid_bootstrap_draws": len(bootstrap), "observed_test_n": observed}


def holm(p_values):
    require(all(isinstance(p, (float, int)) and math.isfinite(p) and 0 <= p <= 1
                for p in p_values), "Holm p-value range")
    adjusted = [None] * len(p_values)
    floor = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=lambda i: (p_values[i], i))):
        floor = max(floor, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, floor)
    return adjusted


def replay(config, endpoints, canonical, test_ids, methods, frozen, result):
    """Replay scorer RNG schedule, then independently recompute 14 primary rows.

    Calibration and endpoint draws are consumed in the same registered order;
    otherwise the primary bootstrap/sign-flip seed would describe a new draw.
    """
    draws, seed = config["draws"], config["rng_seed"]
    require(type(draws) is int and draws >= 100 and type(seed) is int, "registered RNG")
    n = len(test_ids)
    require(n > 0 and set(endpoints) == set(frozen["prior_normalization"]), "test/denominator panel")
    model_order = list(config["models"])
    method_order = ["decoder1200"] + [f"{model}__{arm}" for model in model_order
                                     for arm in ("neutral", "t0")] + ["full1400"]
    method_order += [f"{key}__{suffix}" for key in frozen["selected"]
                     for suffix in ("fixed_half", "cal_selected")]
    require(set(method_order) == set(methods) and len(method_order) == len(methods) == 14,
            "registered method order/set")
    labels = {endpoint: np.asarray([canonical[rid].get(endpoint, -1) for rid in test_ids], dtype=int)
              for endpoint in endpoints}
    rng = np.random.default_rng(seed)
    # The scorer draws calibration intervals before contrast uncertainty.
    for method in method_order:
        for endpoint in endpoints:
            count = int(np.count_nonzero(labels[endpoint] >= 0))
            require(count > 0, "empty calibration endpoint")
            rng.integers(count, size=(draws, count))

    reported = {(r["family"], r["system"], r["reference"]): r
                for r in result["primary_aggregate"]}
    contrasts = [(r["family"], r["system"], r["reference"])
                 for r in config["test_contrasts"]]
    require(len(contrasts) == len(set(contrasts)) == len(reported) == 14
            and set(contrasts) == set(reported), "14 registered primary contrasts")
    endpoint_reported = {(r["family"], r["system"], r["reference"], r["endpoint"]): r
                         for r in result["results"]}
    require(len(endpoint_reported) == len(endpoints) * len(contrasts), "endpoint result grid")
    summaries = []
    for family, system, reference in contrasts:
        values = {}
        for endpoint, spec in endpoints.items():
            y = labels[endpoint]
            a = losses(methods[system][endpoint], y, spec["kind"])
            b = losses(methods[reference][endpoint], y, spec["kind"])
            require(len(a) == len(b) == np.count_nonzero(y >= 0) and len(a) >= 2,
                    "paired endpoint labels")
            with np.errstate(invalid="ignore"):
                diff = a - b
            full = np.full(n, np.nan)
            full[y >= 0] = diff
            values[endpoint] = full
            endpoint_row = endpoint_reported[family, system, reference, endpoint]
            if np.isfinite(diff).all():
                indices = rng.integers(len(diff), size=(draws, len(diff)))
                ci = np.quantile(diff[indices].mean(axis=1), [.025, .975]).tolist()
                signs = rng.choice([-1, 1], size=(draws, len(diff)))
                p = float((1 + np.count_nonzero(np.abs((signs * diff).mean(axis=1))
                                                  >= abs(diff.mean()))) / (draws + 1))
            else:
                ci, p = None, 1.0
            same(ci, endpoint_row["paired_people_ci95"], "endpoint bootstrap")
            same(p, endpoint_row["signflip_p"], "endpoint sign flip")
        recalculated = aggregate(values,
            {key: frozen["prior_normalization"][key]["cal_prior_mean_primary_loss"]
             for key in endpoints},
            {key: spec["family"] for key, spec in endpoints.items()}, rng, draws)
        record = reported[family, system, reference]
        for key in ("estimate", "paired_people_ci95_conditional_valid",
                    "paired_signflip_p", "failed_bootstrap_fraction"):
            same(recalculated[key], record[key], f"{family}/{system}: {key}")
        require(recalculated["status"] == record["status"]
                and recalculated["observed_test_n"] == record["observed_test_n"]
                and recalculated.get("valid_bootstrap_draws") == record.get("valid_bootstrap_draws"),
                "aggregate status, denominators or valid draw count")
        summaries.append({"family": family, "system": system, "reference": reference,
                          "estimate": recalculated["estimate"],
                          "paired_people_ci95_conditional_valid":
                              recalculated["paired_people_ci95_conditional_valid"],
                          "paired_signflip_p": recalculated["paired_signflip_p"],
                          "failed_bootstrap_fraction": recalculated["failed_bootstrap_fraction"],
                          "valid_bootstrap_draws": recalculated.get("valid_bootstrap_draws"),
                          "status": recalculated["status"]})
    for family in ("main", "t0_vs_neutral"):
        subset = [row for row in summaries if row["family"] == family]
        adjusted = holm([row["paired_signflip_p"] if row["paired_signflip_p"] is not None
                         else 1.0 for row in subset])
        for row, value in zip(subset, adjusted):
            same(value, reported[row["family"], row["system"], row["reference"]]
                 ["holm_p_primary_within_family"], "primary Holm")
            row["holm_p_primary_within_family"] = value
    return summaries
