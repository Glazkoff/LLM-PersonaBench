"""Offline native-engine integration fixture; never scientific evidence."""
import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from aamas2027.native_gigaevo import NativeGigaEvo
from aamas2027.pipeline import neutral_genotype
from src.optim.budget import BudgetLedger
from src.optim.fitness import FitnessResult
from src.optim.genotype import to_json
from src.optim.results import CellWriter, atomic_write_json


def schema_value(schema, root, code, key=''):
    if '$ref' in schema:
        node = root
        for p in schema['$ref'].split('/')[1:]:
            node = node[p]
        return schema_value(node, root, code, key)
    if 'enum' in schema:
        return schema['enum'][0]
    if 'anyOf' in schema:
        return schema_value(next(s for s in schema['anyOf'] if s.get('type') != 'null'), root, code, key)
    t = schema.get('type')
    if t == 'object' or 'properties' in schema:
        return {k: schema_value(v, root, code, k) for k, v in schema.get('properties', {}).items()}
    if t == 'array':
        return []
    if t in ('integer', 'number'):
        return max(1, schema.get('minimum', 1))
    if t == 'boolean':
        return True
    return code if key == 'code' else 'Fixture-only evidence.'


class FixtureFitness:
    forward_passes_per_candidate = 12

    def evaluate(self, g):
        self.score = .1 if 'IMPROVED_FIXTURE' in g['role_definition'] else .4
        return FitnessResult(self.score, self.score, -.5, 1., 120, .001, (0., 0.), self.score)

    def last_descriptor(self):
        return dict(prompt_length_chars=3000, predicted_sd=1.)


def run(output):
    output = Path(output)
    seed = neutral_genotype()
    improved = {**seed, 'role_definition': seed['role_definition'] + ' IMPROVED_FIXTURE'}
    requests = []
    class Mock(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(payload)
            tools = payload.get('tools', [])
            rf = payload.get('response_format') or {}
            schema = tools[0]['function']['parameters'] if tools else rf.get('json_schema', {}).get('schema', {})
            value = schema_value(schema, schema, to_json(improved)) if schema else {'insights': []}
            message = {'role': 'assistant', 'content': json.dumps(value)}
            finish = 'stop'
            if tools:
                message.update(content=None, tool_calls=[dict(id='fixture_call', type='function', function=dict(name=tools[0]['function']['name'], arguments=json.dumps(value)))])
                finish = 'tool_calls'
            data = dict(id='fixture', object='chat.completion', created=1, model='fixture-mutator',
                        choices=[dict(index=0, message=message, finish_reason=finish)],
                        usage=dict(prompt_tokens=100, completion_tokens=200, total_tokens=300))
            b = json.dumps(data).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Mock)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    os.environ['MUTATOR_BASE_URL'] = f'http://127.0.0.1:{server.server_port}/v1'
    try:
        writer = CellWriter(output, 'native_fixture')
        arm = NativeGigaEvo(dict(max_mutants=3, timeout_s=180), np.random.default_rng(1), SimpleNamespace(model='fixture-mutator'), seed)
        ledger = BudgetLedger(3)
        selected = arm.run(FixtureFitness(), seed, ledger, writer)
        assert 'IMPROVED_FIXTURE' in selected['role_definition']
        assert 24 <= ledger.forward_passes <= 36
        assert requests
        atomic_write_json(output / 'NATIVE_FIXTURE_PASSED.json', dict(scientific_evidence=False,
            label='synthetic integration fixture', native_mutator_requests=len(requests), budget=ledger.to_dict()))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.output)
