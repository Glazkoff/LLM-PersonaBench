import csv
import json

import pytest

from aamas2027 import external as E


def _fixture(tmp_path):
    catalog = [
        {"QuestionID": "QID10", "QuestionText": "History", "QuestionType": "MC",
         "Options": ["A", "B"], "Settings": {"Selector": "SAVR"}, "csv_columns": ["QID10"]},
        {"QuestionID": "QID20", "QuestionText": "Repeated target", "QuestionType": "MC",
         "Options": ["A", "B"], "Settings": {"Selector": "SAVR"}, "csv_columns": ["QID20"]},
        {"QuestionID": "QID30", "QuestionText": "Choice", "QuestionType": "MC",
         "Options": ["A", "B", "C"], "Settings": {"Selector": "SAVR"}, "csv_columns": ["QID30"]},
        {"QuestionID": "QID40", "QuestionText": "Agreement", "QuestionType": "Matrix",
         "Columns": ["Disagree", "Neutral", "Agree"], "Settings": {"Selector": "Likert"}, "csv_columns": ["QID40_1"]},
    ]
    (tmp_path / "question_catalog.json").write_text(json.dumps(catalog))
    for name, fields in (("wave1_3_response.csv", ["pid", "QID10", "QID20", "QID30", "QID40_1"]),
                         ("wave4_response.csv", ["pid", "QID20", "QID30", "QID40_1"])):
        with (tmp_path / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for pid in range(1, 8):
                writer.writerow({key: (str(pid) if key == "pid" else "1") for key in fields})
    receipt = {"dataset": E.DATASET, "revision": "a" * 40,
               "sha256": {name: E.digest(tmp_path / name) for name in E.FILES}}
    (tmp_path / "download_receipt.json").write_text(json.dumps(receipt))
    spec = {"source": "catalog_csv", "overlap_audit": True,
            "families": sorted(E.FAMILIES), "equivalent_question_ids": [],
            "targets": [{"column": "QID30", "kind": "nominal", "family": "economic_preferences"},
                        {"column": "QID40_1", "kind": "ordinal", "family": "heuristics_cognitive_bias"}]}
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(spec))
    return path, spec


def test_prepare_excludes_target_questions_and_types_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "SPLIT_SIZES", (("fit", 2), ("dev", 1), ("cal", 1)))
    selection, _ = _fixture(tmp_path)
    result = E.prepare_twin(tmp_path, selection, tmp_path / "out")
    assert result["allowed_history_columns"] == ["QID10", "QID20"]
    assert [x["metric"] for x in result["target_catalog"]] == ["brier", "rps"]
    assert result["target_observed_n"] == {"QID30": 7, "QID40_1": 7}
    assert len(result["split"]["test"]) == 3
    rows = [json.loads(s) for s in (tmp_path / "out/rows.jsonl").read_text().splitlines()]
    assert all("QID30" not in r["history"] and "QID40_1" not in r["history"] for r in rows)
    assert all(r["respondent_id"].startswith("twin2k:") for r in rows)


def test_equivalent_question_and_unaudited_input_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "SPLIT_SIZES", (("fit", 2), ("dev", 1), ("cal", 1)))
    selection, spec = _fixture(tmp_path)
    spec["equivalent_question_ids"] = ["QID20"]
    selection.write_text(json.dumps(spec))
    result = E.prepare_twin(tmp_path, selection, tmp_path / "out")
    assert result["allowed_history_columns"] == ["QID10"]
    spec["overlap_audit"] = False
    selection.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="audit"):
        E.prepare_twin(tmp_path, selection, tmp_path / "bad")
    spec["overlap_audit"] = True
    spec["full_persona"] = True
    selection.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="forbidden"):
        E.prepare_twin(tmp_path, selection, tmp_path / "bad")


def test_nominal_cannot_be_scored_as_ordinal(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "SPLIT_SIZES", (("fit", 2), ("dev", 1), ("cal", 1)))
    selection, spec = _fixture(tmp_path)
    spec["targets"][0]["kind"] = "ordinal"
    selection.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Likert"):
        E.prepare_twin(tmp_path, selection, tmp_path / "out")


def test_t0_component_only_and_tsmall_gate():
    export = E.t0_instruction_export({"role_definition": "Answer as this respondent.",
                                      "critic_formulations": "Use observed answers.",
                                      "facet_formulations": {"N1": "high"}})
    assert export["method_transfer_claim"] is False
    assert "facet_formulations" not in export
    with pytest.raises(ValueError, match="gated"):
        E.tsmall_schedule(["a", "b"], adapter_ready=True)
    cells = E.tsmall_schedule(["a", "b"], adapter_ready=True, readout_probed=True)
    assert len(cells) == 12 and {x["budget"] for x in cells} == {40}


def test_preparation_requires_pinned_matching_receipt_and_immutable_output(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "SPLIT_SIZES", (("fit", 2), ("dev", 1), ("cal", 1)))
    selection, _ = _fixture(tmp_path)
    receipt = tmp_path / "download_receipt.json"
    saved = receipt.read_text()
    receipt.unlink()
    with pytest.raises(ValueError, match="receipt required"):
        E.prepare_twin(tmp_path, selection, tmp_path / "out")
    receipt.write_text(saved)
    with (tmp_path / "wave4_response.csv").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="hash differs"):
        E.prepare_twin(tmp_path, selection, tmp_path / "out")
    receipt_data = json.loads(saved)
    receipt_data["sha256"]["wave4_response.csv"] = E.digest(tmp_path / "wave4_response.csv")
    receipt.write_text(json.dumps(receipt_data))
    E.prepare_twin(tmp_path, selection, tmp_path / "out")
    with pytest.raises(FileExistsError, match="already exist"):
        E.prepare_twin(tmp_path, selection, tmp_path / "out")
