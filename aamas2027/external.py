"""Fail-closed preparation of external PersonBench transfer data.

This module runs on CPU. It never sends human records to a model or selects an
endpoint from model outcomes. The prepared rows remain local to the data host.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from urllib.request import urlopen

DATASET = "LLM-Digital-Twin/Twin-2K-500"
SUBDIR = "question_catalog_and_human_response_csv"
FILES = ("question_catalog.json", "wave1_3_response.csv", "wave4_response.csv")
FAMILIES = {"economic_preferences", "heuristics_cognitive_bias"}
METRICS = {"ordinal": "rps", "nominal": "brier", "binary": "brier"}
FORBIDDEN = {"full_persona", "persona_summary", "persona_text", "persona_json", "wave4_Q_wave1_3_A"}
SPLIT_SIZES = (("fit", 1000), ("dev", 200), ("cal", 200))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_catalog_csv(out, *, revision):
    """Fetch only the official catalog and two numeric CSVs, never full personas."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("pin the dataset to a 40-character commit revision")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / "download_receipt.json"
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text())
        if prior.get("revision") != revision or prior.get("dataset") != DATASET:
            raise ValueError("download directory belongs to another dataset revision")
        for name in FILES:
            if not (out / name).is_file() or digest(out / name) != prior.get("sha256", {}).get(name):
                raise ValueError(f"cached file differs from pinned receipt: {name}")
        return prior
    if any((out / name).exists() for name in FILES):
        raise ValueError("unreceipted files in download directory; use a clean directory")
    hashes = {}
    for name in FILES:
        path = out / name
        if not path.exists():
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{SUBDIR}/{name}"
            with urlopen(url, timeout=120) as response, path.with_suffix(path.suffix + ".part").open("wb") as target:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    target.write(block)
            path.with_suffix(path.suffix + ".part").replace(path)
        hashes[name] = digest(path)
    receipt = {"dataset": DATASET, "revision": revision, "sha256": hashes,
               "source": f"https://huggingface.co/datasets/{DATASET}"}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def _rows(path):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or reader.fieldnames[0] != "pid" or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"invalid pid/header in {path}")
        rows = list(reader)
    ids = [r["pid"] for r in rows]
    if any(not x or not re.fullmatch(r"[0-9]+", x) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"missing, nonnumeric or duplicate pid in {path}")
    return reader.fieldnames, rows


def _catalog(path):
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, list):
        raise ValueError("catalog must be a list")
    by_col = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("QuestionID"), str):
            raise ValueError("malformed catalog entry")
        for col in entry.get("csv_columns", []):
            if col in by_col:
                # QIDs can be repeated but standardized CSV columns must not be.
                raise ValueError(f"duplicate catalog column {col}")
            by_col[col] = entry
    return entries, by_col


def _target(entry, column, declaration):
    if declaration.get("column") != column:
        raise ValueError("target column mismatch")
    kind = declaration.get("kind")
    if kind not in METRICS:
        raise ValueError("target kind requires explicit ordinal, nominal, or binary declaration")
    if column not in entry.get("csv_columns", []):
        raise ValueError("target absent from catalog")
    qtype = entry.get("QuestionType")
    selector = entry.get("Settings", {}).get("Selector")
    if kind == "ordinal":
        if qtype != "Matrix" or selector != "Likert" or not entry.get("Columns"):
            raise ValueError("ordinal RPS requires documented Likert scale")
        options = entry["Columns"]
    elif kind in {"nominal", "binary"}:
        if qtype != "MC" or selector not in {"SAVR", "SAHR"} or not entry.get("Options"):
            raise ValueError("nominal Brier requires single-choice MC")
        options = entry["Options"]
        if kind == "binary" and len(options) != 2:
            raise ValueError("binary target needs two options")
    if len(options) < 2 or len(set(map(str, options))) != len(options):
        raise ValueError("target options are missing or duplicated")
    return {"column": column, "question_id": entry["QuestionID"], "question_text": entry.get("QuestionText"),
            "block": entry.get("BlockName"), "kind": kind, "metric": METRICS[kind], "options": options,
            "family": declaration["family"]}


def prepare_twin(data_dir, selection_path, out_dir):
    """Write a provenance receipt, disjoint split and leakage-filtered JSONL rows.

    selection JSON declares target columns, their type, family, and all QIDs
    judged equivalent/paraphrased. Human audit must be explicitly recorded.
    """
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    if (out_dir / "rows.jsonl").exists() or (out_dir / "manifest.json").exists():
        raise FileExistsError("prepared rows or manifest already exist; use a new output directory")
    source_receipt = data_dir / "download_receipt.json"
    if not source_receipt.is_file():
        raise ValueError("pinned download receipt required before preparation")
    receipt = json.loads(source_receipt.read_text())
    if receipt.get("dataset") != DATASET or not re.fullmatch(r"[0-9a-f]{40}", receipt.get("revision", "")):
        raise ValueError("invalid pinned source receipt")
    if any(digest(data_dir / name) != receipt.get("sha256", {}).get(name) for name in FILES):
        raise ValueError("source file hash differs from pinned receipt")
    selection = json.loads(Path(selection_path).read_text())
    if selection.get("source") != "catalog_csv" or selection.get("overlap_audit") is not True:
        raise ValueError("catalog_csv source and completed target-equivalence audit required")
    if set(selection.get("families", [])) != FAMILIES:
        raise ValueError("both prespecified families must be present")
    targets = selection.get("targets", [])
    if not targets or any(t.get("family") not in FAMILIES for t in targets):
        raise ValueError("declare target columns from both families")
    if {t["family"] for t in targets} != FAMILIES:
        raise ValueError("targets must cover both families")
    if any(k in json.dumps(selection) for k in FORBIDDEN):
        raise ValueError("forbidden full persona or repeated-answer source")
    h1, r1 = _rows(data_dir / FILES[1])
    h4, r4 = _rows(data_dir / FILES[2])
    if set(r["pid"] for r in r1) != set(r["pid"] for r in r4):
        raise ValueError("wave pid sets differ")
    _, by_col = _catalog(data_dir / FILES[0])
    target_cols = [t["column"] for t in targets]
    if len(set(target_cols)) != len(target_cols) or any(c not in h4 or c not in by_col for c in target_cols):
        raise ValueError("duplicate or unknown target column")
    typed = [_target(by_col[t["column"]], t["column"], t) for t in targets]
    excluded_qids = {x["question_id"] for x in typed}
    extra = selection.get("equivalent_question_ids", [])
    if not isinstance(extra, list) or any(not isinstance(q, str) for q in extra):
        raise ValueError("equivalent_question_ids must be a list")
    excluded_qids.update(extra)
    known_qids = {e["QuestionID"] for e in json.loads((data_dir / FILES[0]).read_text())}
    if not excluded_qids <= known_qids:
        raise ValueError("unknown equivalent question ID")
    # Unknown columns could hide repeats. No inference from numeric names.
    if any(c not in by_col for c in h1[1:]):
        raise ValueError("wave1_3 has columns absent from catalog")
    allowed = [c for c in h1[1:] if by_col[c]["QuestionID"] not in excluded_qids]
    if not allowed:
        raise ValueError("no allowed history remains")
    ids = sorted(set(r["pid"] for r in r1), key=int)
    if len(ids) <= sum(n for _, n in SPLIT_SIZES):
        raise ValueError("insufficient respondents for fixed split")
    random.Random(20261004).shuffle(ids)
    split, start = {}, 0
    for name, n in SPLIT_SIZES:
        split[name] = [f"twin2k:{pid}" for pid in ids[start:start+n]]
        start += n
    split["test"] = [f"twin2k:{pid}" for pid in ids[start:]]
    out_dir.mkdir(parents=True, exist_ok=True)
    by_pid1, by_pid4 = ({r["pid"]: r for r in rows} for rows in (r1, r4))
    membership = {pid.split(":", 1)[1]: name for name, group in split.items() for pid in group}
    observed = {c: 0 for c in target_cols}
    with (out_dir / "rows.jsonl").open("w") as f:
        for pid in ids:
            targets_now = {c: by_pid4[pid][c] for c in target_cols if by_pid4[pid][c].strip()}
            for c, raw in targets_now.items():
                try:
                    value = int(raw)
                except ValueError as exc:
                    raise ValueError(f"non-categorical target {c}: {raw!r}") from exc
                if not 1 <= value <= len(next(t["options"] for t in typed if t["column"] == c)):
                    raise ValueError(f"target value outside declared options: {c}")
                observed[c] += 1
            row = {"respondent_id": f"twin2k:{pid}", "split": membership[pid],
                   "history": {c: by_pid1[pid][c] for c in allowed if by_pid1[pid][c].strip()},
                   "targets": targets_now}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"version": "twin-catalog-csv-v1", "source": DATASET,
                "source_revision": receipt["revision"],
                "selection_sha256": digest(selection_path),
                "source_sha256": {name: digest(data_dir / name) for name in FILES},
                "split": split, "target_catalog": typed, "allowed_history_columns": allowed,
                "excluded_question_ids": sorted(excluded_qids), "target_observed_n": observed,
                "rows_sha256": digest(out_dir / "rows.jsonl"),
                "status": "adapter_prepared_readout_unprobed"}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def tsmall_schedule(models, *, adapter_ready=False, readout_probed=False):
    if not adapter_ready or not readout_probed:
        raise ValueError("Tsmall gated until adapter data and readout are probed")
    if len(models) != 2 or len(set(models)) != 2:
        raise ValueError("freeze exactly two distinct models")
    return [{"model": model, "arm": arm, "seed": seed, "budget": 40,
             "representation": "neutral_domain_valid_history", "label_access": "external_dev"}
            for model in models for arm in ("ga", "best_of_b") for seed in (0, 1, 2)]


def t0_instruction_export(source_genotype):
    """Export only portable instruction text, never invented NEO facet values."""
    required = ("role_definition", "critic_formulations")
    if any(not isinstance(source_genotype.get(k), str) or not source_genotype[k].strip() for k in required):
        raise ValueError("source genotype lacks portable instruction components")
    return {"track": "T0", "representation": "observed_history_only",
            "method_transfer_claim": False, "source_target_labels_used": False,
            "role_definition": source_genotype["role_definition"],
            "critic_formulations": source_genotype["critic_formulations"],
            "omitted_components": ["trait_formulations", "facet_formulations", "intensity_modifiers"]}


def load_ipip300(path, key_path, *, short_observed_numbers):
    """Read recoded IPIP-300 with a namespaced ID and explicit 60/180 map.

    The 300-row corpus has been used in development. Values outside 1..5 are
    missing, and never enter calibration as an observed label.
    """
    import pandas as pd

    key = pd.read_excel(key_path)
    if len(key) != 300 or set(key["Full#"].astype(int)) != set(range(1, 301)):
        raise ValueError("IPIP-300 key must list Full# 1..300")
    short_rows = key.dropna(subset=["Short#"])
    if len(short_rows) != 120 or set(short_rows["Short#"].astype(int)) != set(range(1, 121)):
        raise ValueError("IPIP short map must be bijective 1..120")
    short_map = dict(zip(short_rows["Short#"].astype(int), short_rows["Full#"].astype(int)))
    observed = list(short_observed_numbers)
    if len(observed) != 60 or len(set(observed)) != 60 or not set(observed) <= set(short_map):
        raise ValueError("matched B1 inputs must be 60 distinct short-form items")
    full_observed = [short_map[n] for n in observed]
    targets = sorted(set(range(1, 301)) - set(short_map.values()))
    if len(targets) != 180:
        raise ValueError("IPIP-300 transfer target map is not 180 items")
    rows = json.loads(Path(path).read_text())
    if len(rows) != 300 or len(set(str(r["case"]) for r in rows)) != 300:
        raise ValueError("IPIP-300 requires 300 unique local cases")
    all_values = []
    for row in rows:
        values = []
        for i in range(1, 301):
            raw = row.get(f"i{i}")
            try:
                val = float(raw)
            except (TypeError, ValueError):
                val = float("nan")
            values.append(int(val) if val in (1, 2, 3, 4, 5) else None)
        all_values.append({"respondent_id": f"palign-ipip300:{row['case']}",
                           "observed_60": {str(i): values[i-1] for i in full_observed},
                           "targets_180": {str(i): values[i-1] for i in targets}})
    return {"source": "PAlign/Test-set.json", "status": "development_exposed_descriptive",
            "input_full_numbers": full_observed, "target_full_numbers": targets,
            "reverse_keyed_targets": [int(r["Full#"]) for _, r in key.iterrows()
                                      if int(r["Full#"]) in targets and str(r["Sign"]).strip().startswith("-")],
            "rows": all_values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download")
    d.add_argument("--out", required=True)
    d.add_argument("--revision", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--data", required=True)
    a.add_argument("--selection", required=True)
    a.add_argument("--out", required=True)
    r = sub.add_parser("readiness")
    r.add_argument("--data", required=True)
    r.add_argument("--selection")
    args = p.parse_args()
    if args.command == "download":
        result = download_catalog_csv(args.out, revision=args.revision)
    elif args.command == "prepare":
        result = prepare_twin(args.data, args.selection, args.out)
    else:
        present = {name: (Path(args.data) / name).is_file() for name in FILES}
        result = {"files_present": present, "selection_present": bool(args.selection and Path(args.selection).is_file()),
                  "download_inputs_present": all(present.values()), "data_gate": False,
                  "next": "pin source, freeze and audit target selection, then run prepare",
                  "readout_probed": False, "tsmall_ready": False}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
