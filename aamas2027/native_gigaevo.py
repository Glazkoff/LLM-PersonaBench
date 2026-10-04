"""Native internal GigaEvo engine, with a non-executable JSON evaluator bridge.

The private engine is an explicit dependency, not vendored into the public paper
source. Its SHA and exact CLI are saved. Its native guided-mutation pipeline,
MAP-Elites archive and scheduler execute unchanged; JSON task prompt adaptation
and an external forecast-cap stopper are disclosed adapter components.
"""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import yaml

from src.optim.arms.base import Arm
from src.optim.arms.gepa_official import strict_genotype
from src.optim.genotype import parse_genotype, to_json
from src.optim.results import atomic_write_json

PIN = 'f72ab08c628cc49f4c4c1b6ee48bd11b68d3ce08'


class NativeGigaEvo(Arm):
    name = 'gigaevo_native'

    def search(self, seed_genotype):
        source = Path(os.environ['HBS_GIGAEVO_SOURCE']).resolve()
        interpreter = os.environ['HBS_GIGAEVO_PYTHON']
        pinfile = source / 'SOURCE_SHA'
        pin = pinfile.read_text().strip() if pinfile.exists() else subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if pin != PIN:
            raise RuntimeError('native GigaEvo source does not match pinned commit')
        problem = self.writer.dir / 'native_problem'
        native_output = self.writer.dir / 'native_output'
        problem.mkdir(exist_ok=True)
        (problem / 'initial_programs').mkdir(exist_ok=True)
        (problem / 'initial_programs/seed.json').write_text(to_json(seed_genotype))
        (problem / 'task_description.txt').write_text(
            'Improve a personality simulation prompt represented as a JSON object. The code field must be a JSON document, never executable source. '
            'Preserve exactly the role_definition, trait_formulations, facet_formulations, critic_formulations keys. '
            'Preserve all existing 5 trait names and 30 facet names. Only change description text. Respondent intensity values are inserted by a fixed renderer. '
            'Maximize fitness=1-RPS on an optimization panel. No test or selection answers are available.\nSeed schema:\n' + to_json(seed_genotype))
        specs = {}
        for name, bounds, primary in [('fitness', [0, 1], True), ('is_valid', [0, 1], False),
                                       ('prompt_length_chars', [0, 16000], False), ('predicted_sd', [0, 2], False)]:
            specs[name] = dict(description=name, decimals=6, is_primary=primary, higher_is_better=True,
                               lower_bound=bounds[0], upper_bound=bounds[1], include_in_prompts=True,
                               significant_change=.001, sentinel_value=-1000 if primary else 0)
        (problem / 'metrics.yaml').write_text(yaml.safe_dump(dict(specs=specs)))
        # Only representation-specific instructions change; all guided evidence
        # and mutation mechanisms remain the engine's source-pinned defaults.
        promptdir = problem / 'prompts/mutation'
        promptdir.mkdir(parents=True, exist_ok=True)
        text = (source / 'gigaevo/prompts/mutation/system.txt').read_text()
        text = text.replace('Python programs', 'JSON persona documents').replace('Python program', 'JSON persona document')
        text = text.replace('`code` is Python source, not JSON** — no embedded JSON, templates, or format examples.',
                            '`code` is the complete JSON persona document** — never Python source or Markdown fences.')
        (promptdir / 'system.txt').write_text(text)
        lock, stopflag = threading.Lock(), problem / 'FORECAST_BUDGET_REACHED'
        owner = self
        proxy_calls = []
        fatal_errors = []
        seed = int(self.rng.integers(2**31 - 1))
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                body = json.dumps({'object': 'list', 'data': [{'id': owner.mutator.model, 'object': 'model'}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                try:
                    if self.path == '/fitness':
                        with lock:
                            if owner.ledger.remaining < 1:
                                result = dict(fitness=-1000, is_valid=0, prompt_length_chars=0, predicted_sd=0)
                                stopflag.touch()
                            else:
                                try:
                                    g = strict_genotype(json.dumps(payload), seed_genotype)
                                except (ValueError, KeyError):
                                    owner.ledger.invalid_proposals += 1
                                    result = dict(fitness=-1000, is_valid=0, prompt_length_chars=0, predicted_sd=0)
                                else:
                                    _, score = owner._eval(g, len(owner.history), [], 'native_validator')
                                    desc = owner.fitness.last_descriptor()
                                    result = dict(fitness=1-score, is_valid=1, **desc)
                                owner.ledger.proposal_attempts += 1
                                if owner.ledger.remaining < 1:
                                    stopflag.touch()
                    elif self.path == '/v1/chat/completions':
                        import urllib.request
                        with lock:
                            call_id = len(proxy_calls)
                            proxy_calls.append(dict(call_id=call_id, state='requested'))
                        payload.update(seed=seed + call_id, temperature=.7, max_tokens=int(owner.cfg.get('max_tokens', 2500)))
                        payload['chat_template_kwargs'] = {'enable_thinking': False}
                        url = os.environ['MUTATOR_BASE_URL'].rstrip('/') + '/chat/completions'
                        request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                            headers={'Authorization': 'Bearer EMPTY', 'Content-Type': 'application/json'})
                        with urllib.request.urlopen(request, timeout=900) as response:
                            result = json.load(response)
                        usage = result.get('usage') or {}
                        with lock:
                            owner.ledger.charge_fraction(0, prompt_tokens=usage.get('prompt_tokens', 0), generated_tokens=usage.get('completion_tokens', 0))
                            owner.ledger.mutator_calls += 1
                            proxy_calls[call_id].update(state='completed', usage=usage)
                            atomic_write_json(owner.writer.dir / 'native_mutator_cost.json', proxy_calls)
                    else:
                        raise ValueError('unknown bridge route')
                    body = json.dumps(result).encode()
                    self.send_response(200)
                except Exception as e:
                    fatal_errors.append(repr(e))
                    stopflag.touch()
                    body = json.dumps({'error': repr(e)}).encode()
                    self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = f'http://127.0.0.1:{server.server_port}'
        (problem / 'validate.py').write_text(
            'import json\nimport urllib.request\n\ndef validate(payload):\n'
            f'    req = urllib.request.Request({endpoint + "/fitness"!r}, data=json.dumps(payload).encode(), headers={{"Content-Type":"application/json"}})\n'
            '    with urllib.request.urlopen(req, timeout=3600) as r:\n        return json.load(r)\n')
        env = {**os.environ, 'OPENAI_API_KEY': 'EMPTY', 'LOCAL_LLM_PROXY': endpoint + '/v1',
               'PYTHONPATH': str(Path(__file__).resolve().parents[1]) + os.pathsep + str(source),
               'HBS_NATIVE_SEED': str(seed), 'WANDB_MODE': 'disabled', 'LANGFUSE_TRACING_ENABLED': 'false'}
        launcher = problem / 'launch.py'
        launcher.write_text('import os, random, runpy\nimport numpy as np\ns=int(os.environ["HBS_NATIVE_SEED"])\nrandom.seed(s)\nnp.random.seed(s)\n' +
                            f'runpy.run_path({str(source / "run.py")!r}, run_name="__main__")\n')
        args = [interpreter, str(launcher), 'problem.name=hbs_persona', f'problem.dir={problem}',
                'program_format=json_document', 'pipeline=guided', 'memory=none', 'llm=local_proxy',
                'algorithm=single_island_no_distant_parents', 'storage=disk',
                f'model_name={self.mutator.model}', 'temperature=0.7', f'max_tokens={self.cfg.get("max_tokens", 2500)}',
                'max_in_flight=1', 'max_concurrent_dags=1', 'dag_concurrency=1', 'llm_max_concurrent=1',
                'prefetch_factor=1', 'sync_min_delta=1', 'runner_poll_interval=0.1', 'loop_interval=0.1',
                'post_cap_drain_grace_s=120', f'max_mutants={self.cfg.get("max_mutants", 160)}',
                'stopper._target_=aamas2027.native_stop.ForecastStopper', f'+stopper.flag={stopflag}',
                'behavior_space.keys=[prompt_length_chars,predicted_sd]',
                'behavior_space.bounds=[[0,16000],[0,2]]', 'behavior_space.resolutions=[8,8]',
                'behavior_space.binning_types=[linear,linear]', f'hydra.run.dir={native_output}']
        atomic_write_json(self.writer.dir / 'native_provenance.json', dict(source_sha=pin, args=args,
            identity='native GigaEvo internal 2.20.1', adapter_changes=['JSON-format prompt adaptation', 'external forecast stopper', 'shared validator'],
            evaluator='same fixed global optimization panel and raw RPS'))
        try:
            with open(self.writer.dir / 'native_engine.log', 'w') as log:
                proc = subprocess.run(args, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=float(self.cfg.get('timeout_s', 36000)))
            if proc.returncode:
                raise RuntimeError(f'native GigaEvo exited {proc.returncode}; see native_engine.log')
            if fatal_errors:
                raise RuntimeError(f'native bridge failed: {fatal_errors}')
            if self.best_genotype is None or len(self.history) < 2:
                raise RuntimeError('native engine did not evaluate seed plus a mutant')
            if any(c['state'] != 'completed' for c in proxy_calls):
                raise RuntimeError('native mutator has unaccounted failed requests')
            return self.best_genotype
        finally:
            server.shutdown()
            server.server_close()
            atomic_write_json(self.writer.dir / 'native_mutator_cost.json', proxy_calls)
