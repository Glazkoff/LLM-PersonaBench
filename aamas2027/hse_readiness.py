"""Read-only HSE readiness audit inside a finite allocation; no scientific claims."""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lock', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--gpu', action='store_true')
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('finite Slurm allocation required')
    report = dict(scientific_evidence=False, job=os.environ['SLURM_JOB_ID'],
                  host=platform.node(), models={}, packages={}, errors=[])
    for name in ('torch', 'vllm', 'numpy', 'pandas', 'scipy', 'scikit-learn',
                 'openai', 'transformers', 'gepa', 'gigaevo'):
        try:
            report['packages'][name] = md.version(name)
        except md.PackageNotFoundError:
            report['packages'][name] = None
    for name, revision in json.loads(Path(a.lock).read_text()).items():
        root = Path('/home/lsavchenko/.cache/huggingface/hub') / (
            'models--' + name.replace('/', '--')) / 'snapshots' / revision
        try:
            config = json.loads((root / 'config.json').read_text())
            index = json.loads((root / 'model.safetensors.index.json').read_text())
            shards = sorted(set(index['weight_map'].values()))
            missing = [s for s in shards if not (root / s).is_file() or (root / s).stat().st_size == 0]
            report['models'][name] = dict(revision=revision, path=str(root),
                model_type=config.get('model_type'), shard_count=len(shards),
                shard_bytes=sum((root / s).stat().st_size for s in shards if (root / s).is_file()),
                missing=missing, config_sha256=hashlib.sha256((root / 'config.json').read_bytes()).hexdigest())
            if missing:
                report['errors'].append(f'{name}: missing shards')
        except Exception as e:
            report['errors'].append(f'{name}: {type(e).__name__}: {e}')
    if a.gpu:
        try:
            import torch
            torch.manual_seed(20261005)
            report['gpu_witness'] = []
            for i in range(torch.cuda.device_count()):
                x = torch.randn(128, 128, device=f'cuda:{i}')
                y = x @ x.T
                torch.cuda.synchronize(i)
                assert y.isfinite().all().item()
                report['gpu_witness'].append(dict(device=i, name=torch.cuda.get_device_name(i),
                    shape=list(y.shape), finite=True))
            if len(report['gpu_witness']) != 2:
                report['errors'].append('expected two assigned GPUs')
            import vllm
            report['vllm_import'] = True
        except Exception as e:
            report['errors'].append(f'GPU stack: {type(e).__name__}: {e}')
    else:
        corpus = Path('/home/lsavchenko/personality-arr/LLM-PersonaBench/data/raw/df_ipipneo_120_clusters')
        h = hashlib.sha256()
        with corpus.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        report['corpus_sha256'] = h.hexdigest()
    report['check_passed'] = not report['errors']
    report['campaign_ready'] = False  # Still needs matched environments and real-model parity.
    Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if report['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
