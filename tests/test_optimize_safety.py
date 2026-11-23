# SPDX-License-Identifier: MIT
import hashlib
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'scripts'))
import optimize
from fedbrain.runner import RunConfig,atomic_json


class OptimizeSafetyTests(unittest.TestCase):
    def test_completed_run_requires_same_configuration_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            src=root/'fedbrain'/'example.py'
            src.parent.mkdir()
            src.write_text('# frozen source\n',encoding='utf-8')
            cfg={'seed':42}
            old={'status':'completed','config':asdict(RunConfig(**cfg)),
                 'source_hashes':{'example.py':hashlib.sha256(src.read_bytes()).hexdigest()}}
            atomic_json(root/'runs'/'fake'/'result.json',old)
            with patch.object(optimize,'ROOT',root),patch.object(optimize,'run_training',side_effect=AssertionError('must not train')):
                self.assertEqual(optimize.run_or_reuse('fake',cfg),old)
                with self.assertRaisesRegex(RuntimeError,'different requested configuration'):
                    optimize.run_or_reuse('fake',{'seed':43})
                src.write_text('# changed source\n',encoding='utf-8')
                with self.assertRaisesRegex(RuntimeError,'fingerprints'):
                    optimize.run_or_reuse('fake',cfg)

    def test_stale_in_memory_selection_cannot_open_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'selection.json'
            locked={'selected_variant':'E','selected_run_id':'opt-E-s42','test_was_not_evaluated_for_selection':True}
            atomic_json(path,locked)
            stale={**locked,'selected_variant':'D'}
            with patch.object(optimize,'SELECTION',path):
                with self.assertRaisesRegex(RuntimeError,'Exact validation decision'):
                    optimize.finalize_locked_test('opt-E-s42',stale)

    def test_search_reuses_lock_without_reopening_finalized_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'selection.json'
            locked={'variants':[{'id':'A','config':optimize.configuration('A',42,True)}]}
            atomic_json(path,locked)
            with patch.object(optimize,'SELECTION',path),patch.object(optimize,'run_or_reuse',side_effect=AssertionError('must not reopen test')):
                self.assertEqual(optimize.search(),locked)

    def test_checkpoint_fingerprint_checked_before_loading_or_test_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run=root/'runs'/'opt-E-s42'
            run.mkdir(parents=True)
            (run/'checkpoint.pt').write_bytes(b'changed untrusted bytes')
            atomic_json(run/'result.json',{'test_evaluated':False})
            path=root/'selection.json'
            locked={'selected_run_id':'opt-E-s42','test_was_not_evaluated_for_selection':True,
                    'checkpoint_fingerprints':{'opt-E-s42':'invalidsha'}}
            atomic_json(path,locked)
            with patch.object(optimize,'ROOT',root),patch.object(optimize,'SELECTION',path),patch.object(optimize,'load_subjects',side_effect=AssertionError('must not open subjects')):
                with self.assertRaisesRegex(RuntimeError,'pre-test fingerprint'):
                    optimize.finalize_locked_test('opt-E-s42',locked)


if __name__=='__main__':
    unittest.main()
