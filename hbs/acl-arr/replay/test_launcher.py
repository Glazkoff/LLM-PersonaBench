import copy, json, os, shutil, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import launcher as L
from types import SimpleNamespace

def fixture(path):
    mapping=json.loads((L.PACKAGE/'replay/cells.json').read_text())
    needed={'replay/cells.json'}|{c['source'] for c in mapping['cells']}|{c['manuscript'] for c in mapping['cells']}
    for name in needed:
        target=path/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(L.PACKAGE/name,target)
    return path

class Guards(unittest.TestCase):
    def test_mapped_cells(self):
        self.assertGreaterEqual(L.cells(L.PACKAGE)['checked_cells'],29)
    def test_mutated_aggregate_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p=fixture(Path(d)/'pkg')
            f=p/'aggregates/controls.json';r=json.loads(f.read_text());r['absolute_losses']['decoder1200']['normalized']+=.01
            f.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'source cell changed'):L.cells(p)
    def test_mutated_manuscript_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p=fixture(Path(d)/'pkg')
            f=p/'paper/main.tex';f.write_text(f.read_text().replace('decoder1200 & .910600','decoder1200 & .999999'))
            with self.assertRaisesRegex(ValueError,'manuscript cell missing'):L.cells(p)
    def test_no_slurm_denied(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'requires Slurm'):L.finite_cpu()
    def test_gpu_denied(self):
        with patch.dict(os.environ,{'SLURM_JOB_ID':'123'}),patch.object(L.subprocess,'check_output',return_value='TimeLimit=00:10:00 AllocTRES=cpu=4,gres/gpu=1'):
            with self.assertRaisesRegex(ValueError,'CPU-only'):L.finite_cpu()
    def test_unlimited_denied(self):
        with patch.dict(os.environ,{'SLURM_JOB_ID':'123'}),patch.object(L.subprocess,'check_output',return_value='TimeLimit=UNLIMITED AllocTRES=cpu=4'):
            with self.assertRaisesRegex(ValueError,'finite'):L.finite_cpu()
    def test_plan_only_does_not_execute(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);gate=root/'ops-20261106/postprobe-r2-review-2';gate.mkdir(parents=True)
            for name in ('bundle-20261106-stimulus-r2','twin-20261106-stimulus-r2'):(root/name).mkdir()
            (gate/'verify_readout.py').write_text('# fixture, never imported\n')
            (gate/'COMPARE_SCORE_VERIFIED.json').write_text('{}')
            with patch.object(L,'GATE',L.sha(gate/'verify_readout.py')),patch.object(L.subprocess,'run') as runner:
                result=L.restricted(SimpleNamespace(root=str(root),out=str(root/'new'),mode='r2',execute=False))
                self.assertFalse(result['execute']);runner.assert_not_called();self.assertFalse((root/'new').exists())
    def test_output_overwrite_denied(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'output exists'):
                L.restricted(SimpleNamespace(root=d,out=d,mode='controls',execute=True))
if __name__=='__main__':unittest.main()
