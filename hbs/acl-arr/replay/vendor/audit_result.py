"""Independent R2 point-estimate and uncertainty readback after score gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import audit_uncertainty as U

FAMILIES = ("economic_preferences", "heuristics_cognitive_bias")
ARMS = ("neutral", "t0")
MODELS = ("granite", "qwen")
CONTRAST_N = 14
ENDPOINT_N = 49
PRICE_N = 40
PERSON_N = 2058


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def same(a, b, label, tol=1e-9):
    if a is None or b is None:
        require(a is b, f"{label}: null/finite mismatch")
    else:
        require(isinstance(a, (float, int)) and isinstance(b, (float, int))
                and math.isfinite(a) and math.isfinite(b)
                and math.isclose(float(a), float(b), rel_tol=tol, abs_tol=1e-10),
                f"{label}: numerical mismatch")


def load_gate(path, expected_sha):
    path = Path(path).resolve(strict=True)
    require(path.name == "verify_readout.py", "registered read-only R2 gate script required")
    require(re.fullmatch(r"[0-9a-f]{64}", expected_sha or "") is not None
            and sha(path) == expected_sha, "approved R2 score-gate source hash drift")
    spec = importlib.util.spec_from_file_location("r2_postscore_gate", path)
    require(spec is not None and spec.loader is not None, "cannot load R2 gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def admitted(gate, gate_path, bundle, lane, upstream, gate_receipt):
    allocation = gate.require_finite_cpu_slurm()
    require(Path(gate_receipt).is_file(), "verified compare-score receipt absent")
    prior = read(gate_receipt)
    current = gate.verify_comparison(bundle, lane, upstream, "compare-score")
    require(prior.get("status") == "compare_score_complete_exploratory"
            and {k: v for k, v in prior.items() if k != "verification_allocation"} == current,
            "compare-score receipt differs from current successful R2 Slurm/source gate")
    return allocation, current, {"gate_script_sha256": sha(gate_path),
                                 "score_gate_receipt_sha256": sha(gate_receipt)}


def align_person(source_row, by_endpoint, endpoints, price_keys):
    require({by_endpoint[key]["source_endpoint_id"] for key in price_keys} == price_keys
            and all(by_endpoint[key]["source_endpoint_id"] == key
                    for key in set(endpoints) - price_keys),
            "source slots do not form an exact product permutation")
    aligned = {}
    for endpoint, spec in endpoints.items():
        slot = by_endpoint[endpoint]["source_endpoint_id"]
        raw = source_row[slot].strip()
        if not raw:
            continue
        require(raw.isdigit() and 1 <= int(raw) <= len(spec["options"]),
                "canonical source response outside endpoint option order")
        aligned[endpoint] = int(raw) - 1
    return aligned


def source_labels(lane, protocol, manifest, amendment):
    """Rebuild product-keyed labels from source CSV slot cells and locked question sidecar."""
    prepared = lane / "private/prepared"
    data = lane / "private/data"
    csv_path = data / "wave4_response.csv"
    require(sha(csv_path) == manifest["source_sha256"]["wave4_response.csv"],
            "source Wave4 CSV changed")
    endpoints = {e["column"]: e for e in protocol["endpoints"]}
    require(len(endpoints) == ENDPOINT_N and {e["family"] for e in endpoints.values()} == set(FAMILIES),
            "registered endpoint set drift")
    membership = {person: split for split, people in manifest["split"].items() for person in people}
    require(len(membership) == PERSON_N and set(manifest["split"]) == {"fit", "dev", "cal", "test"},
            "registered respondent split drift")
    raw_rows = {}
    with (prepared / "rows.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            rid = row["respondent_id"]
            require(rid in membership and rid not in raw_rows and row["split"] == membership[rid],
                    "prepared raw row identity drift")
            raw_rows[rid] = row
    require(set(raw_rows) == set(membership), "prepared raw rows incomplete")
    stimuli = {rid: {} for rid in membership}
    with (prepared / "STIMULI.jsonl").open() as stream:
        for line in stream:
            item = json.loads(line)
            rid, endpoint = item["respondent_id"], item["endpoint_id"]
            require(rid in membership and endpoint in endpoints
                    and item["split"] == membership[rid] and endpoint not in stimuli[rid]
                    and item["options"] == endpoints[endpoint]["options"],
                    "source stimulus identity or ordered options drift")
            stimuli[rid][endpoint] = item
    require(all(set(by_endpoint) == set(endpoints) for by_endpoint in stimuli.values()),
            "source stimulus grid incomplete")
    price_keys = {key for key, spec in endpoints.items() if spec["family"] == FAMILIES[0]}
    require(len(price_keys) == PRICE_N, "canonical purchase product count changed")
    csv_rows = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and reader.fieldnames[0] == "pid"
                and len(reader.fieldnames) == len(set(reader.fieldnames))
                and set(endpoints) <= set(reader.fieldnames),
                "Wave4 numeric CSV source-column header drift")
        for row in reader:
            rid = "twin2k:" + row["pid"]
            require(rid in membership and rid not in csv_rows, "Wave4 CSV person identity drift")
            csv_rows[rid] = row
    require(set(csv_rows) == set(membership), "Wave4 CSV person set differs from frozen split")
    canonical = {}
    counts = {split: {endpoint: 0 for endpoint in endpoints} for split in manifest["split"]}
    for rid in membership:
        csv_row = csv_rows[rid]
        prepared_slot = raw_rows[rid]["targets"]
        for slot in endpoints:
            value = csv_row[slot].strip()
            require(prepared_slot.get(slot, "").strip() == value,
                    "prepared raw source slot differs from pinned Wave4 CSV")
        aligned = align_person(csv_row, stimuli[rid], endpoints, price_keys)
        for endpoint in aligned:
            counts[membership[rid]][endpoint] += 1
        canonical[rid] = aligned
    require(amendment["response_column_semantics"] == "presentation_slot"
            and counts == amendment["canonical_target_observed_n"],
            "independently reconstructed canonical label counts differ from frozen amendment")
    return endpoints, canonical, counts


def forecasts(spec, ids, endpoints, split):
    path = Path(spec["path"])
    require(sha(path) == spec["sha256"] and spec["endpoint_options"] ==
            {key: value["options"] for key, value in endpoints.items()},
            "forecast SHA or option order differs from config")
    result = {endpoint: [] for endpoint in endpoints}
    expected = ((rid, endpoint) for rid in ids for endpoint in endpoints)
    n = 0
    with path.open() as stream:
        for line in stream:
            rid, endpoint = next(expected, (None, None))
            record = json.loads(line)
            p = record.get("probabilities")
            require(set(record) == {"respondent_id", "endpoint_id", "split", "probabilities"}
                    and record["respondent_id"] == rid and record["endpoint_id"] == endpoint
                    and record["split"] == split
                    and isinstance(p, list) and len(p) == len(endpoints[endpoint]["options"])
                    and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in p)
                    and abs(sum(p) - 1) <= 1e-8,
                    "forecast identity, ordered options or probability simplex drift")
            result[endpoint].append(tuple(float(x) for x in p))
            n += 1
    require(n == len(ids) * len(endpoints), "forecast grid incomplete")
    return result


def proper_loss(p, y, kind):
    k = len(p)
    if kind == "ordinal":
        cumulative = 0.0
        error = 0.0
        for threshold in range(k - 1):
            cumulative += p[threshold]
            error += (cumulative - (1.0 if y <= threshold else 0.0)) ** 2
        primary = error / (k - 1)
    else:
        primary = -math.log(p[y]) if p[y] > 0 else math.inf
    brier = sum((value - (1.0 if index == y else 0.0)) ** 2
                for index, value in enumerate(p)) / k
    return primary, brier


def holm(values):
    require(all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                for value in values), "reported p-value outside [0,1]")
    m = len(values)
    result = [None] * m
    running = 0.0
    for rank, index in enumerate(sorted(range(m), key=lambda i: (values[i], i))):
        running = max(running, (m - rank) * values[index])
        result[index] = min(1.0, running)
    return result


def checked_result(bundle, lane, gate_receipt, gate_script, gate_script_sha256):
    gate = load_gate(gate_script, gate_script_sha256)
    allocation, score_gate, gate_sources = admitted(gate, gate_script, bundle, lane,
                                                     lane.parent / "twin-20261106-v6", gate_receipt)
    protocol = read(bundle / "external/TWIN_PROTOCOL_FROZEN.json")
    manifest = read(lane / "private/prepared/manifest.json")
    amendment = read(bundle / "external/TWIN_STIMULUS_AMENDMENT.json")
    endpoints, canonical, counts = source_labels(lane, protocol, manifest, amendment)
    compare = lane / "private/compare"
    config_path, frozen_path, result_path = [compare / name for name in
        ("TEST_CONFIG.json", "FROZEN.json", "RESULT.json")]
    config, frozen, result = map(read, (config_path, frozen_path, result_path))
    require(score_gate["scored_result_sha256"] == sha(result_path)
            and config["mixture_freeze_sha256"] == sha(frozen_path)
            and result["status"] == "external_test_scored_exploratory"
            and result["config_sha256"] == sha(config_path)
            and result["mixture_freeze_sha256"] == sha(frozen_path)
            and frozen["stimulus_lineage"] == result["stimulus_lineage"]
            and frozen["stimulus_lineage"] == gate.stimulus_binding(bundle),
            "verified score/mixture/sidecar chain changed after gate")
    ids = manifest["split"]
    test_ids = ids["test"]
    methods = {"decoder1200": forecasts(config["decoder1200"]["test"], test_ids, endpoints, "test"),
               "full1400": forecasts(config["full1400"]["test"], test_ids, endpoints, "test")}
    for model in MODELS:
        for arm in ARMS:
            methods[f"{model}__{arm}"] = forecasts(config["models"][model][arm]["test"],
                                                   test_ids, endpoints, "test")
    selected = frozen["selected"]
    require(set(selected) == {f"{model}__{arm}" for model in MODELS for arm in ARMS},
            "CAL frozen model/arm set drift")
    for key, by_endpoint in selected.items():
        require(set(by_endpoint) == set(endpoints), "CAL frozen endpoint set drift")
        for suffix in ("fixed_half", "cal_selected"):
            methods[f"{key}__{suffix}"] = {}
            for endpoint in endpoints:
                weight = 0.5 if suffix == "fixed_half" else by_endpoint[endpoint]["weight_persona"]
                require(type(weight) in (float, int) and weight in frozen["grid"],
                        "CAL mixture weight outside registered grid")
                a = methods[key][endpoint]
                b = methods["decoder1200"][endpoint]
                methods[f"{key}__{suffix}"][endpoint] = [tuple(weight * x + (1 - weight) * z
                    for x, z in zip(pa, pb)) for pa, pb in zip(a, b)]
    require(len(methods) == CONTRAST_N, "method family changed")
    return gate, allocation, gate_sources, protocol, manifest, endpoints, canonical, counts, config, frozen, result, methods


def audit(bundle, lane, gate_receipt, gate_script, gate_script_sha256,
          uncertainty_source_sha256):
    require(re.fullmatch(r"[0-9a-f]{64}", uncertainty_source_sha256 or "") is not None
            and sha(Path(U.__file__)) == uncertainty_source_sha256,
            "independent uncertainty source hash drift")
    (gate, allocation, gate_sources, protocol, manifest, endpoints, canonical, counts,
     config, frozen, result, methods) = checked_result(bundle, lane, gate_receipt,
                                                       gate_script, gate_script_sha256)
    ids = manifest["split"]
    # Independently reconstruct the frozen CAL denominator from FIT labels.
    denominators = {}
    for endpoint, spec in endpoints.items():
        k = len(spec["options"])
        observed_fit = [canonical[rid][endpoint] for rid in ids["fit"] if endpoint in canonical[rid]]
        count = [observed_fit.count(choice) for choice in range(k)]
        prior = [(n + 1) / (len(observed_fit) + k) for n in count]
        cal_y = [canonical[rid][endpoint] for rid in ids["cal"] if endpoint in canonical[rid]]
        denominator = sum(proper_loss(prior, y, spec["kind"])[0] for y in cal_y) / len(cal_y)
        record = frozen["prior_normalization"][endpoint]
        require(record["fit_observed_n"] == len(observed_fit)
                and record["cal_observed_n"] == len(cal_y)
                and record["family"] == spec["family"] and record["kind"] == spec["kind"]
                and record["metric"] == spec["primary_metric"]
                and record["fit1000_laplace_alpha"] == 1
                and len(record["fit1000_prior_probabilities"]) == k,
                "CAL normalization cohort/type drift")
        for a, b in zip(prior, record["fit1000_prior_probabilities"]):
            same(a, b, "CAL prior probability")
        same(denominator, record["cal_prior_mean_primary_loss"], "CAL prior loss")
        require(denominator > 1e-6, "CAL prior denominator unavailable")
        denominators[endpoint] = denominator
    require(set(frozen["prior_normalization"]) == set(endpoints), "CAL denominator set incomplete")
    # Recompute every frozen CAL grid score without calling the comparator's
    # scoring code or using TEST labels to choose a mixture.
    cal_ids = ids["cal"]
    cal_base = forecasts(config["decoder1200"]["cal"], cal_ids, endpoints, "cal")
    require(frozen["grid"] == [round(i / 10, 1) for i in range(11)],
            "CAL mixture grid differs from preregistered eleven weights")
    for model in MODELS:
        for arm in ARMS:
            key = f"{model}__{arm}"
            cal_persona = forecasts(config["models"][model][arm]["cal"],
                                    cal_ids, endpoints, "cal")
            for endpoint, spec in endpoints.items():
                observed = [(i, canonical[rid][endpoint]) for i, rid in enumerate(cal_ids)
                            if endpoint in canonical[rid]]
                scores = []
                for weight in frozen["grid"]:
                    losses = []
                    for i, y in observed:
                        p = tuple(weight * a + (1 - weight) * b for a, b in zip(
                            cal_persona[endpoint][i], cal_base[endpoint][i]))
                        losses.append(proper_loss(p, y, spec["kind"])[0])
                    scores.append(sum(losses) / len(losses) if all(math.isfinite(x) for x in losses)
                                  else None)
                choice = frozen["selected"][key][endpoint]
                require(choice["cal_labeled_n"] == len(observed)
                        and len(choice["cal_scores"]) == len(scores)
                        and choice["cal_nonfinite_weights"] ==
                            [frozen["grid"][i] for i, value in enumerate(scores) if value is None],
                        "CAL frozen mixture endpoint denominator/grid drift")
                for actual, reported in zip(scores, choice["cal_scores"]):
                    same(actual, reported, "CAL mixture proper-loss grid")
                eligible = [i for i, value in enumerate(scores) if value is not None]
                require(eligible, "CAL mixture has no finite candidate")
                winner = min(eligible, key=lambda i: (scores[i], frozen["grid"][i]))
                same(frozen["grid"][winner], choice["weight_persona"],
                     "CAL selected mixture weight")
    contrasts = [(c["family"], c["system"], c["reference"]) for c in config["test_contrasts"]]
    registered = set()
    for model in MODELS:
        for arm in ARMS:
            key = f"{model}__{arm}"
            registered.update(("main", method, "full1400") for method in
                              (key, f"{key}__fixed_half", f"{key}__cal_selected"))
        registered.add(("t0_vs_neutral", f"{model}__t0", f"{model}__neutral"))
    require(len(contrasts) == CONTRAST_N and len(set(contrasts)) == CONTRAST_N,
            "registered contrast set incomplete")
    require(set(contrasts) == registered, "registered comparison family changed")
    reported_aggregates = {(r["family"], r["system"], r["reference"]): r
                           for r in result["primary_aggregate"]}
    reported_endpoints = {(r["family"], r["system"], r["reference"], r["endpoint"]): r
                          for r in result["results"]}
    require(len(reported_aggregates) == CONTRAST_N
            and len(reported_endpoints) == CONTRAST_N * ENDPOINT_N
            and set(reported_aggregates) == set(contrasts), "reported aggregate contrast set drift")
    require(set(result["test_label_ledger"]) == set(endpoints),
            "TEST label-denominator endpoint set incomplete")
    require(set(reported_endpoints) == {(f, s, r, e) for f, s, r in contrasts for e in endpoints},
            "reported endpoint contrast grid incomplete")
    absolute = result["absolute_scores"]
    require(set(absolute) == set(methods), "reported absolute-score method set drift")
    for method, by_endpoint in methods.items():
        require(set(absolute[method]) == set(endpoints), "reported absolute-score endpoint set drift")
        for endpoint, probabilities in by_endpoint.items():
            observed = [(i, canonical[rid][endpoint]) for i, rid in enumerate(ids["test"])
                        if endpoint in canonical[rid]]
            losses = [proper_loss(probabilities[i], y, endpoints[endpoint]["kind"]) for i, y in observed]
            primary = [x[0] for x in losses]
            record = absolute[method][endpoint]
            require(record["n_labeled"] == len(observed) and record["metric"] ==
                    endpoints[endpoint]["primary_metric"] and record["infinite_primary"] ==
                    (not all(math.isfinite(x) for x in primary)),
                    "absolute-score denominator/type/nonfinite flag drift")
            same(sum(primary) / len(primary) if all(math.isfinite(x) for x in primary) else None,
                 record["mean_primary"], "absolute primary loss")
            if endpoints[endpoint]["kind"] != "ordinal":
                same(sum(x[1] for x in losses) / len(losses), record["mean_brier_secondary"],
                     "absolute secondary Brier")
            else:
                require(record["mean_brier_secondary"] is None, "ordinal Brier should not be reported")
    aggregate_rows, endpoint_rows = [], []
    for family, system, reference in contrasts:
        record = reported_aggregates[family, system, reference]
        endpoint_effects = {name: [] for name in FAMILIES}
        observed_n = {}
        nonfinite = False
        for endpoint, spec in endpoints.items():
            key = (family, system, reference, endpoint)
            reported = reported_endpoints[key]
            losses = []
            brier_differences = []
            for index, rid in enumerate(ids["test"]):
                if endpoint not in canonical[rid]:
                    continue
                y = canonical[rid][endpoint]
                a, ab = proper_loss(methods[system][endpoint][index], y, spec["kind"])
                b, bb = proper_loss(methods[reference][endpoint][index], y, spec["kind"])
                losses.append(a - b)
                brier_differences.append(ab - bb)
            observed_n[endpoint] = len(losses)
            require(reported["kind"] == spec["kind"] and reported["metric"] == spec["primary_metric"]
                    and reported["n_labeled"] == len(losses)
                    and reported["n_missing_human"] == len(ids["test"]) - len(losses)
                    and result["test_label_ledger"][endpoint]["labeled_people"] == len(losses)
                    and result["test_label_ledger"][endpoint]["missing_human_labels"] ==
                        len(ids["test"]) - len(losses)
                    and result["test_label_ledger"][endpoint]["total_people"] == len(ids["test"])
                    and record["observed_test_n"][endpoint] == len(losses),
                    "test endpoint metric or label denominator drift")
            finite = all(math.isfinite(x) for x in losses)
            same(sum(losses) / len(losses) if finite else None,
                 reported["delta_primary"], "endpoint paired proper-loss delta")
            require(reported["infinite_primary"] == (not finite),
                    "endpoint nonfinite flag drift")
            if spec["kind"] != "ordinal":
                same(sum(brier_differences) / len(brier_differences),
                     reported["delta_brier_secondary"], "endpoint Brier delta")
            else:
                require(reported["delta_brier_secondary"] is None,
                        "ordinal secondary Brier should be null")
            if finite:
                endpoint_effects[spec["family"]].append(sum(losses) / len(losses) /
                                                       denominators[endpoint])
            else:
                nonfinite = True
            endpoint_rows.append({"family": family, "system": system, "reference": reference,
                                  "endpoint": endpoint, "n_labeled": len(losses),
                                  "delta_primary": reported["delta_primary"],
                                  "delta_brier_secondary": reported["delta_brier_secondary"],
                                  "signflip_p": reported["signflip_p"],
                                  "holm_p_secondary_endpoint_family":
                                      reported["holm_p_secondary_endpoint_family"]})
        require(set(record["family_weights"]) == set(FAMILIES)
                and all(record["family_weights"][name] == 0.5 for name in FAMILIES)
                and record["min_test_n_per_endpoint"] == 30,
                "registered family weights or endpoint minimum drift")
        family_means = {name: sum(values) / len(values) if values else None
                        for name, values in endpoint_effects.items()}
        estimate = (sum(family_means.values()) / 2
                    if not nonfinite and all(n >= 30 for n in observed_n.values()) else None)
        same(estimate, record["estimate"], "two-family normalized primary estimate")
        aggregate_rows.append({"family": family, "system": system, "reference": reference,
                               "n_endpoints": len(endpoints),
                               "economic_mean_normalized_delta": family_means[FAMILIES[0]],
                               "heuristics_mean_normalized_delta": family_means[FAMILIES[1]],
                               "primary_estimate": estimate,
                               "paired_people_ci95_conditional_valid":
                                   record["paired_people_ci95_conditional_valid"],
                               "paired_signflip_p": record["paired_signflip_p"],
                               "holm_p_primary_within_family":
                                   record["holm_p_primary_within_family"],
                               "status": record["status"]})
    require(len(aggregate_rows) == CONTRAST_N and len(endpoint_rows) == CONTRAST_N * ENDPOINT_N,
            "aggregate/readback table incomplete")
    for family in ("main", "t0_vs_neutral"):
        primary = [r for r in aggregate_rows if r["family"] == family]
        for row, adjusted in zip(primary, holm([r["paired_signflip_p"]
                                                if r["paired_signflip_p"] is not None else 1.0
                                                for r in primary])):
            same(adjusted, row["holm_p_primary_within_family"], "primary Holm adjustment")
        secondary = [r for r in endpoint_rows if r["family"] == family]
        for row, adjusted in zip(secondary, holm([r["signflip_p"] for r in secondary])):
            same(adjusted, row["holm_p_secondary_endpoint_family"], "endpoint Holm adjustment")
    uncertainty = U.replay(config, endpoints, canonical, ids["test"], methods, frozen, result)
    return {"status": "independent_point_and_uncertainty_pass", "scope":
            "Canonical labels, proper-loss means, CAL denominators, 14 primary point estimates and 686 endpoint deltas independently reconstructed. Registered respondent bootstrap and sign-flip draws, 14 primary confidence intervals/p-values, endpoint confidence intervals/p-values and primary Holm adjustments independently replayed from frozen seed and order.",
            "allocation": allocation, "source": gate_sources | {
                "audit_source_sha256": sha(Path(__file__)),
                "uncertainty_source_sha256": sha(Path(U.__file__)),
                "bundle_manifest_sha256": sha(bundle / "SOURCE_SHA256.txt"),
                "prepared_rows_sha256": sha(lane / "private/prepared/rows.jsonl"),
                "sidecar_sha256": sha(lane / "private/prepared/STIMULI.jsonl"),
                "wave4_csv_sha256": sha(lane / "private/data/wave4_response.csv"),
                "cal_freeze_sha256": sha(lane / "private/compare/FROZEN.json"),
                "test_config_sha256": sha(lane / "private/compare/TEST_CONFIG.json"),
                "result_sha256": sha(lane / "private/compare/RESULT.json")},
            "baseline_receipt_ledger": config["baseline_run"],
            "readout_receipts_sha256": {
                model: {phase: config["readout_receipts"][model][phase]["sha256"]
                        for phase in ("cal", "test")}
                for model in MODELS},
            "canonical_target_observed_n": counts, "primary_contrasts": aggregate_rows,
            "uncertainty_contrasts": uncertainty,
            "endpoint_contrasts": endpoint_rows,
            "cal_prior_denominators": [
                {"endpoint": endpoint, "family": spec["family"], "kind": spec["kind"],
                 "metric": spec["primary_metric"],
                 "fit_observed_n": counts["fit"][endpoint],
                 "cal_observed_n": counts["cal"][endpoint],
                 "test_observed_n": counts["test"][endpoint],
                 "cal_fit1000_laplace_prior_mean_loss": denominators[endpoint]}
                for endpoint, spec in endpoints.items()]}


def write_csv(path, rows, fields):
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{name: json.dumps(row[name], allow_nan=False)
                           if isinstance(row[name], (list, dict)) else row[name]
                           for name in fields} for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "lane", "gate-script", "score-gate", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gate-script-sha256", required=True)
    parser.add_argument("--uncertainty-source-sha256", required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "new aggregate-only output directory required")
    result = audit(args.bundle, args.lane, args.score_gate, args.gate_script,
                   args.gate_script_sha256, args.uncertainty_source_sha256)
    args.out.mkdir(parents=True, exist_ok=False)
    summary = {key: value for key, value in result.items()
               if key not in ("primary_contrasts", "uncertainty_contrasts", "endpoint_contrasts", "cal_prior_denominators")}
    summary["primary_contrasts_csv_sha256"] = None
    summary["endpoint_contrasts_csv_sha256"] = None
    summary["cal_prior_denominators_csv_sha256"] = None
    summary["uncertainty_contrasts_csv_sha256"] = None
    write_csv(args.out / "primary_contrasts.csv", result["primary_contrasts"],
              list(result["primary_contrasts"][0]))
    write_csv(args.out / "endpoint_contrasts.csv", result["endpoint_contrasts"],
              list(result["endpoint_contrasts"][0]))
    write_csv(args.out / "cal_prior_denominators.csv", result["cal_prior_denominators"],
              list(result["cal_prior_denominators"][0]))
    write_csv(args.out / "uncertainty_contrasts.csv", result["uncertainty_contrasts"],
              list(result["uncertainty_contrasts"][0]))
    summary["primary_contrasts_csv_sha256"] = sha(args.out / "primary_contrasts.csv")
    summary["endpoint_contrasts_csv_sha256"] = sha(args.out / "endpoint_contrasts.csv")
    summary["cal_prior_denominators_csv_sha256"] = sha(args.out / "cal_prior_denominators.csv")
    summary["uncertainty_contrasts_csv_sha256"] = sha(args.out / "uncertainty_contrasts.csv")
    (args.out / "AUDIT.json").open("x").write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": summary["status"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
