#!/usr/bin/env python3
"""Compare token-exact serial/MTP runs produced by bench_ornith.py."""
import argparse
import json
from pathlib import Path


def compare(baseline: Path, candidate: Path) -> dict:
    inputs = json.loads((baseline / 'inputs.json').read_text())
    if inputs != json.loads((candidate / 'inputs.json').read_text()):
        raise ValueError('Input token sequences differ')
    base = json.loads((baseline / 'results.json').read_text())
    cand = json.loads((candidate / 'results.json').read_text())
    for run in (base, cand):
        if run.get('error') or run.get('exit_code') != 0:
            raise ValueError('A benchmark did not complete successfully')
    key = lambda t: (t['name'], t['prompt_tokens'], t['rep'])
    refs = {key(t): t for t in base['tests']}
    if len(refs) != len(base['tests']) or len(cand['tests']) != len(refs):
        raise ValueError('Different or duplicate test cases')
    rows = []
    seen = set()
    for test in cand['tests']:
        k = key(test)
        if k in seen or k not in refs:
            raise ValueError('Different or duplicate test cases')
        seen.add(k)
        ref = refs[k]
        if ref['reused'] or test['reused']:
            raise ValueError('Comparison requires uncached prompts')
        a, b = ref['output_ids'], test['output_ids']
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if first is None and len(a) != len(b):
            first = min(len(a), len(b))
        done = test['done'].split()
        accepted, offered = int(done[6]), int(done[7])
        if not 0 <= accepted <= offered:
            raise ValueError('Invalid speculative counters')
        rows.append({'workload': k[0], 'prompt_tokens': k[1], 'repeat': k[2],
                     'identical': a == b, 'first_difference': first,
                     'baseline_tokens': len(a), 'candidate_tokens': len(b),
                     'accepted': accepted, 'offered': offered})
    return {'passed': bool(rows) and all(r['identical'] for r in rows),
            'speculation_exercised': any(r['offered'] for r in rows),
            'acceptance_exercised': any(r['accepted'] for r in rows),
            'rejection_exercised': any(r['offered'] > r['accepted'] for r in rows),
            'tests': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] and result['speculation_exercised'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
