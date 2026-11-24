# SPDX-License-Identifier: MIT
"""Guard the submission requirements reviewers check: SPDX, portability, one entry point."""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from scripts import check_compliance

ROOT = Path(__file__).resolve().parent.parent


class ComplianceTests(unittest.TestCase):
    def test_repository_has_no_compliance_violations(self):
        problems = check_compliance.violations()
        self.assertEqual(problems, [], '\n'.join(problems))

    def test_source_files_declare_an_spdx_identifier(self):
        for relative in ('fedbrain/runner.py', 'fedbrain/server.py', 'web/app.js', 'web/index.html',
                         'scripts/pipeline.py', 'scripts/check_compliance.py', 'tests/test_compliance.py',
                         'Makefile', 'setup.sh', 'setup.ps1', 'requirements.txt', 'README.md'):
            text = (ROOT / relative).read_text(encoding='utf-8')
            self.assertIn('SPDX-License-Identifier: MIT', '\n'.join(text.splitlines()[:6]), relative)

    def test_identifier_count_is_not_zero(self):
        files = check_compliance.tracked()
        declared = [f for f in files
                    if (text := check_compliance.read_text(ROOT / f))
                    and check_compliance.declared(text)]
        self.assertGreater(len(declared), 50)

    def test_machine_path_detection_catches_windows_and_home_paths(self):
        windows_sample = f'{"D"}:{chr(92)}project{chr(92)}data'
        home_sample = f'{chr(47)}Users{chr(47)}someone{chr(47)}project'
        self.assertIsNotNone(check_compliance.MACHINE_PATH.search(windows_sample))
        self.assertIsNotNone(check_compliance.MACHINE_PATH.search(home_sample))
        self.assertIsNone(check_compliance.MACHINE_PATH.search('data/abide'))

    def test_committed_run_records_store_relative_directories(self):
        records = list((ROOT / 'runs').glob('*/result.json'))
        self.assertTrue(records)
        for record in records:
            config = json.loads(record.read_text(encoding='utf-8')).get('config', {})
            for key in ('data_dir', 'run_root'):
                value = config.get(key)
                if isinstance(value, str):
                    self.assertFalse(Path(value).is_absolute(), f'{record}: {key}={value}')
                    self.assertIsNone(check_compliance.MACHINE_PATH.search(value), f'{record}: {key}={value}')

    def test_license_files_are_present_and_mit(self):
        license_text = (ROOT / 'LICENSE').read_text(encoding='utf-8')
        self.assertIn('MIT License', license_text)
        self.assertTrue((ROOT / 'LICENSES' / 'MIT.txt').is_file())
        self.assertIn('license = "MIT"', (ROOT / 'pyproject.toml').read_text(encoding='utf-8'))

    def test_single_pipeline_entry_point_is_available(self):
        completed = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'pipeline.py'), '--help'],
                                   cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for command in ('setup', 'data', 'features', 'demo', 'test', 'verify', 'serve', 'all'):
            self.assertIn(command, completed.stdout)
        self.assertTrue((ROOT / 'Makefile').is_file())
        self.assertTrue((ROOT / 'setup.sh').is_file())

    def test_powershell_wrappers_are_optional_not_required(self):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        self.assertIn('scripts/pipeline.py', readme)
        self.assertNotIn('must run setup.ps1', readme)


if __name__ == '__main__':
    unittest.main()
