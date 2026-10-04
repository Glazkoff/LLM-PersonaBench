"""Finite Slurm-owned two-GPU serving supervisor, bound only to loopback."""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def snapshot(model):
    lock = json.loads((ROOT / 'aamas2027/model_lock.json').read_text())
    path = Path('/home/glazkov/hf_cache/hub') / ('models--' + model.replace('/', '--')) / 'snapshots' / lock[model]
    if not (path / 'config.json').exists():
        raise FileNotFoundError(f'pinned model snapshot absent: {path}')
    return str(path)


def launch(model, device, port, log, mutator=False):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': device, 'HF_HOME': '/home/glazkov/hf_cache',
           'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    args = ['/home/glazkov/personality-twins-arr/vllmenv/bin/python', '-m', 'vllm.entrypoints.openai.api_server',
            '--model', snapshot(model), '--served-model-name', model, '--host', '127.0.0.1', '--port', str(port),
            '--trust-remote-code', '--dtype', 'bfloat16', '--gpu-memory-utilization', '.88',
            '--max-model-len', '16384', '--max-num-seqs', '32' if mutator else '128',
            '--max-num-batched-tokens', '8192', '--enable-prefix-caching', '--disable-log-requests']
    f = open(log, 'w')
    proc = subprocess.Popen(args, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    return proc, f, args


def wait_ready(proc, port, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f'vLLM exited {proc.returncode} before readiness')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError('model-serving startup exceeded 15 minutes')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['smoke', 'search', 'final'])
    p.add_argument('--manifest', required=True)
    p.add_argument('--results', required=True)
    p.add_argument('--index', type=int, required=True)
    a = p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('GPU serving requires a finite Slurm allocation')
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    expected = 1 if a.phase == 'final' else 2
    if len(devices) != expected or not all(devices):
        raise RuntimeError(f'{expected} explicitly assigned GPUs required; got {devices}')
    man = json.loads(Path(a.manifest).read_text())
    cell = man['cells'][a.index]
    logs = Path(a.results) / 'serving' / f'{a.phase}-{a.index}-{os.environ["SLURM_JOB_ID"]}'
    logs.mkdir(parents=True)
    processes = []
    evport, mutport = free_port(), free_port()
    while mutport == evport:
        mutport = free_port()
    try:
        processes.append(launch(cell['hf_id'], devices[0], evport, logs / 'readout.log'))
        if a.phase != 'final':
            processes.append(launch(man['mutator'], devices[1], mutport, logs / 'mutator.log', mutator=True))
        for (proc, _, _), port in zip(processes, [evport, mutport]):
            wait_ready(proc, port)
        env = {**os.environ, 'LOCAL_LLM_BASE_URL': f'http://127.0.0.1:{evport}/v1',
               'MUTATOR_BASE_URL': f'http://127.0.0.1:{mutport}/v1', 'HBS_MODEL_SNAPSHOT': snapshot(cell['hf_id']),
               'HF_HOME': '/home/glazkov/hf_cache', 'HF_HUB_OFFLINE': '1'}
        (logs / 'launch.json').write_text(json.dumps(dict(commands=[c for _, _, c in processes],
            phase=a.phase, cell=cell['cell_id'], slurm_job=os.environ['SLURM_JOB_ID']), indent=1))
        module = 'aamas2027.smoke' if a.phase == 'smoke' else 'aamas2027.runner'
        args = [sys.executable, '-m', module]
        if a.phase != 'smoke':
            args += [a.phase]
        args += ['--manifest', a.manifest, '--results', a.results, '--index', str(a.index),
                 '--old-root', '/home/glazkov/personality-twins-arr/LLM-PersonaBench']
        subprocess.run(args, env=env, check=True)
    finally:
        for proc, f, _ in processes:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc, f, _ in processes:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            f.close()


if __name__ == '__main__':
    main()
