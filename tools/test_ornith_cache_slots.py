#!/usr/bin/env python3
"""CPU-only checks for calibrated cache allocation and profile validation."""
import unittest
from ornith_cache_slots import allocate


class CacheSlotsTest(unittest.TestCase):
    def profile(self):
        rows = []
        for layer in range(48):
            hits = [100 if expert < 64 else 1 for expert in range(256)]
            rows.append(f'ORNITH_CACHE layer={layer} bytes={1000000+layer*128} slots=64 '
                        f'cpu_ms={10000 if layer == 0 else 100} cpu_entries=1000 hits='
                        + ','.join(map(str, hits)))
        return '\n'.join(rows)

    def test_budget_and_cost_with_unequal_strides(self):
        result = allocate(self.profile())
        self.assertLessEqual(result['allocated_bytes'], result['budget_bytes'])
        self.assertTrue(all(32 <= count <= 128 for count in result['slots']))
        self.assertGreater(result['slots'][0], 64)
        self.assertEqual(result['slots'], list(map(int, result['env']['STRATA_ORNITH_LAYER_SLOTS'].split(','))))

    def test_full_floor_preserves_allocation(self):
        self.assertEqual(allocate(self.profile(), 1)['slots'], [64]*48)

    def test_bad_profiles(self):
        profile = self.profile()
        for bad in [profile.replace('layer=0 ', 'layer=1 ', 1),
                    profile.replace('cpu_ms=10000 ', 'cpu_ms=nan ', 1),
                    profile.replace('cpu_entries=1000 ', 'cpu_entries=0 ', 1),
                    profile.replace('slots=64 ', 'slots=256 ', 1),
                    '\n'.join(profile.splitlines()[:-1])]:
            with self.subTest(profile=bad[:90]):
                with self.assertRaises(ValueError):
                    allocate(bad)

    def test_invalid_floor(self):
        for fraction in [0, -1, 1.01, float('nan')]:
            with self.assertRaises(ValueError):
                allocate(self.profile(), fraction)


if __name__ == '__main__':
    unittest.main()
