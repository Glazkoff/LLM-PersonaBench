import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float))
    os.replace(tmp, path)


def cell_id(arm, slug, cluster, seed):
    return f"{arm}__{slug}__c{cluster}__s{seed}"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CellWriter:
    def __init__(self, root, cid):
        self.dir = Path(root) / cid
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cid = cid

    def set_status(self, state, **kw):
        atomic_write_json(self.dir / "status.json", {"state": state, "updated": now(), **kw})

    def write_config(self, cfg):
        atomic_write_json(self.dir / "config.json", cfg)

    def write_meta(self, meta):
        atomic_write_json(self.dir / "run_meta.json", meta)

    def append_history(self, rec):
        with open(self.dir / "history.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def write_best(self, g, rec):
        atomic_write_json(self.dir / "best_genotype.json", {"genotype": g, "record": rec})

    def write_front(self, front):
        atomic_write_json(self.dir / "pareto_front.json", front)

    def finish(self, ledger):
        atomic_write_json(self.dir / "budget.json", ledger.to_dict())

    def write_eval(self, obj):
        p = self.dir / "eval_frozen.json"
        if p.exists():
            raise FileExistsError(f"{p} already written; archive the attempt first")
        atomic_write_json(p, obj)

    def archive_attempt(self):
        n = 1
        while (self.dir / "attempts" / str(n)).exists():
            n += 1
        dst = self.dir / "attempts" / str(n)
        dst.mkdir(parents=True)
        for name in ("history.jsonl", "best_genotype.json", "budget.json", "pareto_front.json",
                     "eval_frozen.json", "run_meta.json", "per_respondent.json", "belief_probs.npz"):
            if (self.dir / name).exists():
                shutil.move(str(self.dir / name), dst / name)
        return dst
