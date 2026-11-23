# SPDX-License-Identifier: MIT
"""Portability guards: BOM-tolerant configs, loud config errors, no machine paths in records."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fedbrain.runner import RunConfig, portable_config, read_json, resolve_recorded_path

ROOT = Path(__file__).resolve().parent.parent


class ConfigPortabilityTests(unittest.TestCase):
    def test_read_json_tolerates_a_byte_order_mark(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_bytes(b'\xef\xbb\xbf' + json.dumps({'rounds': 3}).encode('utf-8'))
            self.assertEqual(read_json(path), {'rounds': 3})

    def test_missing_config_fails_instead_of_silently_using_defaults(self):
        completed = subprocess.run(
            [sys.executable, '-m', 'fedbrain.runner', 'train',
             '--config', 'configs/does-not-exist.json', '--run-id', 'should-not-exist'],
            cwd=str(ROOT), capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('config file not found', completed.stderr + completed.stdout)
        self.assertFalse((ROOT / 'runs' / 'should-not-exist').exists())

    def test_unknown_config_key_fails_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'rounds': 1, 'roundz': 2}), encoding='utf-8')
            completed = subprocess.run(
                [sys.executable, '-m', 'fedbrain.runner', 'train',
                 '--config', str(path), '--run-id', 'should-not-exist'],
                cwd=str(ROOT), capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('unknown config keys', completed.stderr + completed.stdout)

    def test_recorded_config_contains_only_portable_directories(self):
        config = RunConfig(rounds=1, data_dir=str(ROOT / 'data' / 'abide'), run_root=str(ROOT / 'runs'))
        recorded = portable_config(config)
        self.assertEqual(recorded['data_dir'], 'data/abide')
        self.assertEqual(recorded['run_root'], 'runs')
        for value in recorded.values():
            if isinstance(value, str) and ('/' in value or '\\' in value):
                self.assertFalse(Path(value).is_absolute(), value)

    def test_recorded_relative_path_resolves_against_the_repository(self):
        self.assertEqual(resolve_recorded_path('data/abide', ROOT / 'fallback'), ROOT / 'data' / 'abide')
        self.assertEqual(resolve_recorded_path('', ROOT / 'fallback'), ROOT / 'fallback')
        absolute = ROOT / 'missing' / 'outside'
        self.assertEqual(resolve_recorded_path(str(absolute), ROOT / 'fallback'), absolute)

    def test_committed_demo_config_stays_portable(self):
        for name in ('federation_demo.json', 'federation_demo_standalone.json'):
            payload = json.loads((ROOT / 'configs' / name).read_text(encoding='utf-8'))
            for key, value in payload.items():
                if isinstance(value, str) and (':' in value or '\\' in value):
                    self.fail(f'{name}: {key}={value!r} looks like a machine path')

    def test_standalone_config_does_not_depend_on_an_uncommitted_checkpoint(self):
        standalone = json.loads((ROOT / 'configs' / 'federation_demo_standalone.json').read_text(encoding='utf-8'))
        self.assertIsNone(standalone['feature_run_id'])
        self.assertEqual(standalone['rounds'], 8)


if __name__ == '__main__':
    unittest.main()
