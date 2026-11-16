"""Analytical file-based fixtures for descriptive health data, not clinical data."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fedbrain.health_data import HealthDataStore


class HealthDataStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        (self.data / 'rois_aal').mkdir(parents=True)
        self.subjects = []
        for sid, site, label, age, frames, rho in [
                ('h1', 'NYU', 0, 10., 32, .2), ('h2', 'UM', 0, 20., 96, .8),
                ('a1', 'NYU', 1, None, 64, -.4), ('a2', 'UM', 1, 15., 48, -.2)]:
            path = self.data / 'rois_aal' / f'{sid}_rois_aal.1D'
            self.write_series(path, frames, rho)
            self.subjects.append({'id': sid, 'file_id': sid, 'site': site, 'label': label,
                                  'age': age, 'path': str(path)})
        self.manifest()
        self.store = HealthDataStore(self.data, self.root / 'cache')

    @staticmethod
    def write_series(path, frames, rho):
        angle = 2 * np.pi * np.arange(frames) / frames
        values = np.column_stack([np.sin(angle * (i % 13 + 1)) + .2 * np.cos(angle * (i % 7 + 1)) for i in range(116)])
        values[:, 0] = 3 * np.cos(angle)
        values[:, 1] = 2 * (rho * np.cos(angle) + np.sqrt(1 - rho * rho) * np.sin(angle))
        values[:, 2] = 4 * np.cos(2 * angle)
        np.savetxt(path, values)

    def manifest(self):
        (self.data / 'manifest.json').write_text(json.dumps({'version': 1, 'subjects': self.subjects}), encoding='utf-8')

    def test_known_correlation_equal_subject_fisher_mean_and_signed_difference(self):
        overview = self.store.overview()
        self.assertEqual((overview['total_subjects'], overview['selected_subjects'], overview['n_hc'], overview['n_asd']), (4, 4, 2, 2))
        hc = np.asarray(overview['groups']['HC']['mean_fc'])
        asd = np.asarray(overview['groups']['ASD']['mean_fc'])
        expected_hc = np.tanh((np.arctanh(.2) + np.arctanh(.8)) / 2)
        expected_asd = np.tanh((np.arctanh(-.4) + np.arctanh(-.2)) / 2)
        self.assertAlmostEqual(hc[0, 1], expected_hc, delta=1e-6)
        self.assertAlmostEqual(asd[0, 1], expected_asd, delta=1e-6)
        duration_weighted = np.tanh((32 * np.arctanh(.2) + 96 * np.arctanh(.8)) / 128)
        self.assertGreater(abs(hc[0, 1] - duration_weighted), .05)
        difference = np.asarray(overview['difference']['fc'])
        np.testing.assert_allclose(difference, asd - hc)
        self.assertLess(difference[0, 1], 0.)
        np.testing.assert_allclose(np.diag(hc), 1.)
        np.testing.assert_allclose(np.diag(difference), 0.)
        self.assertEqual(next(s for s in overview['subjects'] if s['id'] == 'h1')['label_name'], 'HC')
        self.assertEqual(overview['source']['aggregation'], 'equal-subject Fisher-z mean')
        actual_abs = np.abs(hc[np.triu_indices(116, 1)]).mean()
        self.assertAlmostEqual(overview['groups']['HC']['mean_abs_connectivity'], actual_abs)
        json.dumps(overview, allow_nan=False)

    def test_sites_inclusive_ages_histogram_missing_and_empty_groups(self):
        all_data = self.store.overview()
        self.assertEqual(all_data['age_summary']['missing'], 1)
        self.assertEqual(all_data['age_summary']['range'], [10., 20.])
        self.assertEqual(all_data['age_summary']['median'], 15.)
        self.assertEqual(sum(all_data['age_histogram']['hc']), 2)
        self.assertEqual(sum(all_data['age_histogram']['asd']), 1)
        self.assertEqual(all_data['age_histogram']['missing_asd'], 1)
        self.assertEqual(len(all_data['site_counts']), 6)
        self.assertEqual(sum(row['total'] for row in all_data['site_counts']), 4)
        self.assertEqual(self.store.overview(site='NYU')['selected_subjects'], 2)
        self.assertEqual(self.store.overview(age_min=10, age_max=20)['selected_subjects'], 3)
        self.assertEqual(self.store.overview(age_min=15)['selected_subjects'], 2)
        singleton = self.store.overview(age_min=10, age_max=10)
        self.assertEqual(singleton['selected_subjects'], 1)
        self.assertIsNone(singleton['groups']['ASD']['mean_fc'])
        self.assertIsNone(singleton['groups']['ASD']['mean_abs_connectivity'])
        self.assertIsNone(singleton['difference']['fc'])
        empty = self.store.overview(age_min=100)
        self.assertEqual(empty['selected_subjects'], 0)
        self.assertIsNone(empty['groups']['HC']['mean_fc'])
        self.assertIsNone(empty['age_summary']['median'])
        for kwargs in ({'site': 'unknown'}, {'age_min': -1}, {'age_min': float('nan')},
                       {'age_min': 21, 'age_max': 20}, {'age_max': True}):
            with self.assertRaises(ValueError):
                self.store.overview(**kwargs)

    def test_subject_full_curves_links_and_leave_one_out_reference(self):
        hc = self.store.subject('h1')
        self.assertEqual(hc['n_frames'], 32)
        self.assertAlmostEqual(hc['fc'][0][1], .2, delta=1e-6)
        self.assertEqual(hc['reference']['hc_count'], 1)
        self.assertTrue(hc['reference']['leave_one_out'])
        self.assertAlmostEqual(hc['reference']['healthy_mean_fc'][0][1], .8, delta=1e-6)
        self.assertAlmostEqual(hc['reference']['difference_fc'][0][1], -.6, delta=1e-6)
        self.assertEqual(hc['signals']['frames'], list(range(32)))
        self.assertIn('not seconds', hc['signals']['x_unit'])
        self.assertEqual(len(hc['signals']['series']), 3)
        for curve in hc['signals']['series']:
            self.assertEqual(len(curve['values']), 32)
            self.assertAlmostEqual(np.mean(curve['values']), 0., delta=1e-12)
            self.assertAlmostEqual(np.std(curve['values']), 1., delta=1e-12)
            self.assertEqual(curve['label'], f"AAL-{curve['roi'] + 1:03d}")
        self.assertEqual(len(hc['top_connections']), 30)
        magnitudes = [abs(edge['value']) for edge in hc['top_connections']]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))
        for edge in hc['top_connections']:
            self.assertLess(edge['source'], edge['target'])
            self.assertEqual(edge['value'], hc['fc'][edge['source']][edge['target']])
        asd = self.store.subject('a1')
        self.assertEqual(asd['reference']['hc_count'], 2)
        self.assertFalse(asd['reference']['leave_one_out'])
        with self.assertRaises(KeyError):
            self.store.subject('nonexistent')
        json.dumps(hc, allow_nan=False)

    def test_cache_reuse_raw_source_invalidation_and_cohort_metadata_invalidation(self):
        first = self.store.overview()
        cache = self.root / 'cache' / 'raw-fc-v1.npz'
        mtime = cache.stat().st_mtime_ns
        self.store.overview()
        self.assertEqual(cache.stat().st_mtime_ns, mtime)
        other = HealthDataStore(self.data, self.root / 'cache')
        np.testing.assert_array_equal(other.overview()['groups']['HC']['mean_fc'], first['groups']['HC']['mean_fc'])
        self.assertEqual(cache.stat().st_mtime_ns, mtime)
        path = next(s['path'] for s in self.subjects if s['id'] == 'h1')
        self.write_series(path, 32, -.5)
        updated = self.store.overview()
        self.assertNotEqual(first['source']['cohort_fingerprint'], updated['source']['cohort_fingerprint'])
        self.assertAlmostEqual(self.store.subject('h1')['fc'][0][1], -.5, delta=1e-6)
        self.subjects[0]['age'] = 11.
        self.manifest()
        changed = self.store.overview()
        self.assertEqual(changed['age_summary']['min'], 11.)
        self.assertNotEqual(updated['source']['cohort_fingerprint'], changed['source']['cohort_fingerprint'])

    def test_site_previews_are_actual_fc_and_sampled_real_bold_without_mutation(self):
        for offset, (site, rho) in enumerate((('USM', -.1), ('UCLA', .3), ('KKI', .4), ('Yale', .6))):
            sid = 'extra-' + site
            path = self.data / 'rois_aal' / f'{sid}_rois_aal.1D'
            self.write_series(path, 160 + offset * 16, rho)
            self.subjects.append({'id': sid, 'file_id': sid, 'site': site,
                                  'label': 1 if site == 'Yale' else 0, 'age': 18., 'path': str(path)})
        self.manifest()
        self.store.overview()
        cache = self.root / 'cache' / 'raw-fc-v1.npz'
        cache_mtime = cache.stat().st_mtime_ns
        raw_before = {s['id']: Path(s['path']).read_bytes() for s in self.subjects}
        preview = self.store.site_previews()
        self.assertEqual(preview['total_subjects'], 8)
        self.assertEqual(len(preview['sites']), 6)
        self.assertEqual(sum(site['total'] for site in preview['sites']), 8)
        by_site = {site['id']: site for site in preview['sites']}
        self.assertEqual(by_site['NYU']['example']['subject_id'], 'h1')
        self.assertEqual(by_site['UM']['example']['subject_id'], 'h2')
        self.assertEqual(by_site['Yale']['example']['label_name'], 'ASD')
        matrices = []
        for site in preview['sites']:
            example = site['example']
            detail = self.store.subject(example['subject_id'])
            np.testing.assert_array_equal(example['fc'], detail['fc'])
            self.assertEqual(np.asarray(example['fc']).shape, (116, 116))
            self.assertEqual(example['n_frames'], detail['n_frames'])
            bold = example['bold']
            expected_indices = np.linspace(0, detail['n_frames'] - 1, min(128, detail['n_frames']), dtype=int)
            np.testing.assert_array_equal(bold['frame_indices'], expected_indices)
            np.testing.assert_array_equal(bold['values'], np.asarray(detail['signals']['series'][0]['values'])[expected_indices])
            self.assertEqual(bold['roi'], detail['signals']['series'][0]['roi'])
            self.assertEqual(bold['downsampled'], detail['n_frames'] > 128)
            self.assertEqual(bold['frame_indices'][-1], detail['n_frames'] - 1)
            matrices.append(example['fc'][0][1])
        self.assertEqual(len(set(matrices)), 6)
        self.assertEqual(cache.stat().st_mtime_ns, cache_mtime)
        for subject in self.subjects:
            self.assertEqual(Path(subject['path']).read_bytes(), raw_before[subject['id']])
        json.dumps(preview, allow_nan=False)

    def test_empty_site_previews_are_null_and_corrupt_raw_never_invented(self):
        preview = self.store.site_previews()
        empty = next(site for site in preview['sites'] if site['id'] == 'KKI')
        self.assertEqual((empty['total'], empty['hc'], empty['asd']), (0, 0, 0))
        self.assertIsNone(empty['example'])
        selected = next(s for s in self.subjects if s['id'] == 'h1')
        invalid = np.ones((32, 116))
        invalid[0, 0] = np.nan
        np.savetxt(selected['path'], invalid)
        with self.assertRaises(ValueError):
            self.store.site_previews()

    def test_one_hc_reference_is_null_and_corrupt_input_has_no_fallback(self):
        self.subjects = [s for s in self.subjects if s['id'] in ('h1', 'a1')]
        self.manifest()
        hc = self.store.subject('h1')
        self.assertEqual(hc['reference']['hc_count'], 0)
        self.assertIsNone(hc['reference']['healthy_mean_fc'])
        self.assertIsNone(hc['reference']['difference_fc'])
        values = np.ones((32, 116))
        values[0, 0] = np.nan
        np.savetxt(self.subjects[0]['path'], values)
        with self.assertRaises(ValueError):
            self.store.overview()


if __name__ == '__main__':
    unittest.main()
