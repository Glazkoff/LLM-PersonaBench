import json
from pathlib import Path
import pytest
from aamas2027.analysis import freeze, analyze
from aamas2027.pipeline import canonical_hash


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def fixture(root):
    man = dict(sha256='current', cells=[dict(cell_id='x', hf_id='m', arm='ga', fitness='rps_raw')],
               primary_method='mapelites', seeds=[1])
    write(root / 'manifest.json', man)
    write(root / 'panels.json', dict(sha256='panel', splits=dict(eval=[1, 2])))
    decoder = dict(manifest_sha256='current', panel_sha256='panel')
    decoder['sha256'] = canonical_hash(decoder)
    write(root / 'decoders/FROZEN.json', decoder)
    return man


def test_freeze_rejects_stale_candidate(tmp_path):
    fixture(tmp_path)
    write(tmp_path / 'cells/x/selected.json', dict(manifest_sha256='old', panel_sha256='panel'))
    with pytest.raises(ValueError, match='provenance'):
        freeze(tmp_path, tmp_path / 'manifest.json')
    assert not (tmp_path / 'FROZEN.json').exists()


def test_analysis_rejects_result_from_other_run(tmp_path):
    fixture(tmp_path)
    selection = dict(manifest_sha256='current', panel_sha256='panel', selection_score=.2)
    write(tmp_path / 'cells/x/selected.json', selection)
    freeze(tmp_path, tmp_path / 'manifest.json')
    frozen = json.loads((tmp_path / 'FROZEN.json').read_text())
    write(tmp_path / 'decoders/FINAL.json', dict(panel_sha256='panel', eval_case_ids=[1, 2], frozen_sha256=frozen['decoder_sha256']))
    write(tmp_path / 'cells/x/final.json', dict(manifest_sha256='current', panel_sha256='wrong', selected_sha256=canonical_hash(selection)))
    with pytest.raises(ValueError, match='provenance'):
        analyze(tmp_path, tmp_path / 'manifest.json')
    assert not (tmp_path / 'ANALYSIS.json').exists()
