#!/usr/bin/env python3
"""Check profile validation and normalization without loading a model or CUDA."""
import pathlib
import subprocess
import tempfile
import unittest
from ornith_cache_prior import convert

ROOT = pathlib.Path(__file__).resolve().parents[1]

class PriorTest(unittest.TestCase):
    def test_export_rejects_partial_and_duplicate(self):
        row = lambda i: f'ORNITH_CACHE layer={i} bytes=4096 slots=73 cpu_ms=1 cpu_entries=1 hits=' + ','.join(['1']*256)
        good = '\n'.join(row(i) for i in range(48))
        self.assertTrue(convert(good).startswith('ORNITH_CACHE_V1 48 256'))
        for bad in [good+'\n'+row(0), '\n'.join(row(i) for i in range(47)), good.replace('hits=1,', 'hits=,', 1)]:
            with self.assertRaises(ValueError): convert(bad)

    def test_native_reader(self):
        code = r'''
#include "src/qwen35/ornith_cache_profile.hpp"
#include <cassert>
int main(int argc, char** argv) {
    std::vector<strata::ornith::CachePrior> p; std::string error;
    bool ok = strata::ornith::read_cache_prior(argv[1], {64,128}, 3, 4, 5, 2, 32, p, error);
    if (argc > 2) { assert(!ok && p.empty()); return 0; }
    assert(ok && p.size()==2 && p[0].order[0]==1 && p[0].order[1]==0);
    assert(p[0].score[0]==16 && p[0].score[1]==32 && p[0].score[2]==16);
}
'''
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d); (d/'test.cpp').write_text(code)
            subprocess.run(['g++','-std=c++17','-I',str(ROOT),str(d/'test.cpp'),'-o',str(d/'test')],check=True)
            valid = 'ORNITH_CACHE_V1 2 3 4 5\n64 1 2 1\n128 3 2 1\n'
            for i, content in enumerate([valid, valid+'garbage', valid.replace('64 ', '65 '),
                                         valid.replace('1 2 1','0 0 0'), valid.replace('1 2 1','-1 2 1'),
                                         valid.replace('1 2 1','nan 2 1'), valid[:-3], valid.replace('2 3 4 5','3 3 4 5')]):
                (d/'profile').write_text(content)
                subprocess.run([str(d/'test'),str(d/'profile')]+(['invalid'] if i else []),check=True)

if __name__ == '__main__': unittest.main()
