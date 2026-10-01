import json

from src.optim.checks import readout_guard


def test_readout_guard_refuses_missing_or_mismatched_probe(tmp_path):
    models = tmp_path / "models.yaml"
    models.write_text("defaults:\n  gate_mass_on_scale: 0.5\nmodels:\n- hf_id: model/a\n  slug: a\n  tier: 1\n")
    probes = tmp_path / "probes"
    probes.mkdir()
    output = tmp_path / "validation.json"
    assert readout_guard(models, 1, probes, output) == 1
    (probes / "a.json").write_text(json.dumps({"model": "model/b", "mass_mean": 0.99, "n_requests": 160}))
    assert readout_guard(models, 1, probes, output) == 1
    (probes / "a.json").write_text(json.dumps({"model": "model/a", "mass_mean": 0.99, "n_requests": 160}))
    assert readout_guard(models, 1, probes, output) == 0
    assert json.loads(output.read_text())["all_pass"] is True
