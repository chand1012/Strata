#!/usr/bin/env python3
"""Allocate Ornith cache slots from STRATA_ORNITH_CACHE_PROFILE=1 logs.

The output is a fixed per-layer allocation, not a runtime routing prediction.
The byte budget is the recorded allocation, including each layer's aligned tail.
"""
import argparse
import heapq
import json
import math
import pathlib
import re


def allocate(text, min_fraction=0.5):
    if not 0 < min_fraction <= 1:
        raise ValueError('min_fraction must be in (0, 1]')
    pattern = r'^ORNITH_CACHE layer=(\d+) bytes=(\d+) slots=(\d+) cpu_ms=([^ ]+) cpu_entries=(\d+) hits=([0-9,]+)$'
    records = []
    for match in re.finditer(pattern, text, re.M):
        layer, stride, slots, elapsed, entries, hits = match.groups()
        records.append(dict(layer=int(layer), stride=int(stride), slots=int(slots),
                            elapsed=float(elapsed), entries=int(entries), hits=list(map(int, hits.split(',')))))
    records.sort(key=lambda r: r['layer'])
    if len(records) != 48 or [r['layer'] for r in records] != list(range(48)):
        raise ValueError('expected exactly one profile for each of Ornith’s 48 main layers')
    for r in records:
        if (len(r['hits']) != 256 or sum(r['hits']) == 0 or r['stride'] <= 0 or
                not 1 <= r['slots'] < 256 or not math.isfinite(r['elapsed']) or
                r['elapsed'] <= 0 or r['entries'] <= 0):
            raise ValueError('profile needs valid routing counts and measured CPU costs in every layer')
    def size(i, slots):
        return (records[i]['stride'] * slots + 4096 + 255) // 256 * 256
    before = [r['slots'] for r in records]
    budget = sum(size(i, n) for i, n in enumerate(before))
    counts = [max(1, math.ceil(n * min_fraction)) for n in before]
    upper = [min(255, 2*n) for n in before]
    used = sum(size(i, n) for i, n in enumerate(counts))
    # Add-one smoothing and a per-layer floor temper overfitting to calibration text.
    benefit = [[(h+1) / (sum(r['hits'])+256) * r['elapsed']/r['entries']
                for h in sorted(r['hits'], reverse=True)] for r in records]
    queue = []
    def enqueue(i):
        n = counts[i]
        if n < upper[i]:
            delta = size(i,n+1)-size(i,n)
            heapq.heappush(queue, (-benefit[i][n]/delta, i, delta))
    for i in range(48):
        enqueue(i)
    while queue:
        _, i, delta = heapq.heappop(queue)
        if used + delta <= budget:
            counts[i] += 1
            used += delta
            enqueue(i)
    assert used <= budget
    return {'slots': counts, 'original_slots': before, 'budget_bytes': budget,
            'allocated_bytes': used, 'min_fraction': min_fraction,
            'env': {'STRATA_ORNITH_LAYER_SLOTS': ','.join(map(str,counts))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    parser.add_argument('--min-fraction', type=float, default=0.5)
    args = parser.parse_args()
    result = allocate(args.log.read_text(), args.min_fraction)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
