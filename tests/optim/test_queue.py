import importlib.util
import json

spec = importlib.util.spec_from_file_location("queue_mod", "icml2027/scripts/queue.py")
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)


def test_expand_tier1_counts():
    man = Q.expand("icml2027/configs/grid_tier1.yaml")
    main = [c for c in man["cells"] if c["fitness"] == "rps_cal"]
    abl = [c for c in man["cells"] if c["fitness"] == "s0"]
    import yaml
    g = yaml.safe_load(open("icml2027/configs/grid_tier1.yaml"))
    nmod = sum(1 for m in yaml.safe_load(open("icml2027/configs/models.yaml"))["models"] if m["tier"] in g["models_tier"])
    assert len(main) == len(g["arms"]) * nmod * 4 * 3
    assert len(abl) == 2 * 4 * 3 and all(c["arm"] == "ga" for c in abl)
    ids = [c["cell_id"] for c in man["cells"]]
    assert len(ids) == len(set(ids)) and all(c["budget_B"] == 80 for c in man["cells"])


def test_count_states(tmp_path):
    for cid, s in [("a", "completed"), ("b", "failed_oom"), ("c", "completed")]:
        (tmp_path / cid).mkdir(); (tmp_path / cid / "status.json").write_text(json.dumps({"state": s}))
    assert Q.count_states(tmp_path) == {"completed": 2, "failed_oom": 1}
