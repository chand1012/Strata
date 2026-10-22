#!/usr/bin/env python3
"""Export learned expert identities from an Ornith decode routing profile.

Use calibration text separate from evaluation. The file is quant/layout-specific;
keep its model identity and calibration corpus alongside it. It is not a model.
"""
import argparse
import pathlib
import re


def convert(text):
    pattern = r'^ORNITH_CACHE layer=(\d+) bytes=(\d+) slots=\d+ cpu_ms=\S+ cpu_entries=\d+ hits=([0-9,]+)$'
    rows = [(int(i), int(stride), list(map(int, hits.split(','))))
            for i, stride, hits in re.findall(pattern, text, re.M)]
    rows.sort()
    if [r[0] for r in rows] != list(range(48)):
        raise ValueError('expected one complete profile for each of 48 layers')
    if any(stride <= 0 or len(hits) != 256 or not sum(hits) or max(hits) > 10**15
           for _, stride, hits in rows):
        raise ValueError('invalid stride or expert routing counts')
    return 'ORNITH_CACHE_V1 48 256 3072 1024\n' + ''.join(
        f'{stride} ' + ' '.join(map(str, hits)) + '\n' for _, stride, hits in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output.write_text(convert(args.log.read_text()))


if __name__ == '__main__':
    main()
