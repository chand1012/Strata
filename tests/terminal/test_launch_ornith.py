"""Ornith-1.5-122B (bartowski's IQ3_XS GGUF, two shards) in the Terminal's launch list and setup's own-path table."""
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch.catalog import Catalog
from tests.terminal.launch_support import HW_12, Home, catalog, run_cli

ROOT = Path(__file__).resolve().parents[2]
HW_3090 = {**HW_12, "ram_gb": 61.0, "disk_free_gb": 200.0,
           "gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 3090 Ti", "vram_gb": 24.0, "usable": True, "vendor": "nvidia"}]}


def setup_module_():
    spec = importlib.util.spec_from_file_location("strata_setup_for_test", ROOT / "setup.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class OrnithSetupTable(unittest.TestCase):
    def setUp(self):
        self.s = setup_module_()
        self.fam = self.s.OWN_PATH_FAMILIES["ornith"]

    def test_family_entry(self):
        self.assertEqual(self.fam["arch"], "qwen35moe")
        self.assertEqual(list(self.fam["models"]), ["IQ3_XS"])
        self.assertIs(self.fam["vision"], False)
        self.assertEqual(self.fam["mtp"], 1)
        self.assertIn("bartowski/Ornith-1.5-122B-GGUF", self.s.HF_REVISIONS)

    def test_both_shards_are_listed_with_sizes(self):
        files = self.s.own_path_files(self.fam, "IQ3_XS")
        self.assertEqual(files, ["Ornith-1.5-122B-IQ3_XS-00001-of-00002.gguf", "Ornith-1.5-122B-IQ3_XS-00002-of-00002.gguf"])
        self.assertEqual(sum(self.fam["sha256"][f][0] for f in files), 57864724992)

    def test_single_file_families_unchanged(self):
        q = self.s.OWN_PATH_FAMILIES["qwen36"]
        self.assertEqual(self.s.own_path_files(q, "Q8_0"), ["Qwen3.6-35B-A3B-Q8_0.gguf"])
        self.assertEqual(self.s.own_path_size(q, 64.0), "Q8_0")

    def test_vision_is_off_without_asking(self):
        a = mock.Mock(vision="yes", yes=True)
        self.assertEqual(self.s.own_path_vision(a, self.fam, {"vram_gb": 24.0}, False), "none")


class OrnithLaunch(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda s, fresh=False: dict(HW_3090))
        p.start()
        self.addCleanup(p.stop)

    def test_resolve_and_row(self):
        self.assertEqual(catalog(HW_3090).resolve("iq3_xs", "ornith"), ("ornith", "IQ3_XS"))
        rows = [r for r in catalog(HW_3090).rows() if r.family == "ornith"]
        self.assertEqual([(r.model, r.download_gb, r.images) for r in rows], [("IQ3_XS", 57.9, False)])
        self.assertEqual(catalog(HW_3090).max_context("ornith", "IQ3_XS"), 32768)

    def test_plan(self):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", "--family", "ornith", "--model", "IQ3_XS"])
        self.assertEqual(code, 0, err + out)
        plan = json.loads(out)["plan"]
        self.assertEqual(json.dumps(plan).count("ornith") > 0, True)

    def test_qwen36_still_hidden_from_the_terminal_list(self):
        self.assertFalse([r for r in catalog(HW_3090).rows() if r.family == "qwen36"])


if __name__ == "__main__":
    unittest.main()
