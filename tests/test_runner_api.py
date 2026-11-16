import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from fedbrain.data import SITE_ORDER
from fedbrain.runner import RunConfig, _cap_subjects, _split_with_validation
from fedbrain.server import app


class RunnerTests(unittest.TestCase):
    def test_stratified_smoke_cap_does_not_keep_only_low_id_class(self):
        subjects = [{'id': f'{site}-{i:03d}', 'site': site, 'label': int(i<20)}
                    for site in SITE_ORDER for i in range(60)]
        cfg = RunConfig(max_per_site=20)
        selected = _cap_subjects(subjects, cfg)
        self.assertEqual(len(selected), 120)
        split = _split_with_validation(selected, cfg)
        for site in SITE_ORDER:
            for key in ('train','val','test'):
                self.assertEqual({s['label'] for s in split[key] if s['site']==site}, {0,1})
        sets = [set(s['id'] for s in split[k]) for k in ('train','val','test')]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])

    def test_configuration_rejects_invalid_budget(self):
        for cfg in (RunConfig(rounds=0),RunConfig(noise_scale=-1),RunConfig(method='fake'),RunConfig(homomorphic=True,paillier_bits=256),RunConfig(seed=-1),RunConfig(feature_run_id='../private')):
            with self.assertRaises(ValueError):
                cfg.validate()


class APITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_and_real_status_schema(self):
        self.assertEqual(self.client.get('/api/health').json()['ok'], True)
        result = self.client.get('/api/status').json()
        self.assertEqual(result['paper']['reported_accuracy'],.79)
        self.assertEqual(len(result['dataset']['sites']),6)
        self.assertIn('cuda',result['device'])
        self.assertEqual(result['protocol']['features'],6670)

    def test_optimization_schema_and_validation_only_inference_fence(self):
        result=self.client.get('/api/optimization')
        self.assertEqual(result.status_code,200)
        self.assertIn('status',result.json())
        from fedbrain.runner import ROOT
        if (ROOT/'runs'/'opt-smoke-validation-only'/'checkpoint.pt').is_file():
            response=self.client.get('/api/subjects/50957?run_id=opt-smoke-validation-only')
            self.assertEqual(response.status_code,409)

    def test_federation_legacy_trace_is_not_fabricated(self):
        from fedbrain.runner import ROOT
        if (ROOT/'runs'/'full-fedbrain-s42').is_dir():
            response=self.client.get('/api/runs/full-fedbrain-s42/federation')
            self.assertEqual(response.status_code,200)
            trace=response.json()
            self.assertFalse(trace['available'])
            self.assertEqual(trace['events'],[])
            self.assertEqual(trace['rounds'],[])
            self.assertEqual(self.client.get('/api/runs/full-fedbrain-s42/federation/1').status_code,404)
            self.assertEqual(self.client.get('/api/runs/full-fedbrain-s42/federation/501').status_code,422)
        self.assertEqual(self.client.get('/api/runs/no-such-run/federation').status_code,404)

    def test_real_federation_trace_and_round_api(self):
        from fedbrain.runner import ROOT
        if not (ROOT/'runs'/'federation-demo-s42'/'federation'/'index.json').is_file():
            self.skipTest('Real federation demo not yet recorded')
        response=self.client.get('/api/runs/federation-demo-s42/federation')
        self.assertEqual(response.status_code,200)
        trace=response.json()
        self.assertTrue(trace['available'])
        self.assertEqual(len(trace['site_order']),6)
        completed=trace['completed_rounds']
        self.assertGreater(completed,0)
        frame=self.client.get(f'/api/runs/federation-demo-s42/federation/{completed}').json()
        self.assertEqual(len(frame['clients']),6)
        self.assertEqual(frame['after']['shared_consensus_max_error'],0.)
        self.assertTrue(frame['after']['private_preserved'])
        self.assertEqual(self.client.get('/api/runs/federation-demo-s42/federation/0').status_code,422)

    def test_health_data_does_not_require_a_trained_model(self):
        from fedbrain.runner import ROOT
        if not (ROOT/'artifacts'/'health_data'/'raw-fc-v1.npz').is_file():
            self.skipTest('Real full-series health cache not yet built')
        with patch('fedbrain.server._selected_run',side_effect=AssertionError('Health data must not load a model')):
            response=self.client.get('/api/health-data')
            self.assertEqual(response.status_code,200)
            cohort=response.json()
            self.assertEqual(cohort['total_subjects'],465)
            self.assertEqual((cohort['n_hc'],cohort['n_asd']),(241,224))
            self.assertEqual(len(cohort['groups']['HC']['mean_fc']),116)
            healthy=next(s for s in cohort['subjects'] if s['label']==0)
            detail=self.client.get('/api/health-data/subjects/'+healthy['id'])
            self.assertEqual(detail.status_code,200)
            self.assertEqual(detail.json()['reference']['hc_count'],240)
            self.assertEqual(len(detail.json()['signals']['series']),3)

    def test_site_previews_are_real_data_and_do_not_require_a_model(self):
        from fedbrain.runner import ROOT
        if not (ROOT/'artifacts'/'health_data'/'raw-fc-v1.npz').is_file():
            self.skipTest('Real health-data cache is not available')
        with patch('fedbrain.server._selected_run',side_effect=AssertionError('Site previews must not load a model')):
            response=self.client.get('/api/health-data/sites')
            self.assertEqual(response.status_code,200)
            payload=response.json()
            self.assertEqual(payload['total_subjects'],465)
            self.assertEqual([site['id'] for site in payload['sites']],list(SITE_ORDER))
            self.assertEqual(sum(site['total'] for site in payload['sites']),465)
            self.assertEqual(sum(site['hc'] for site in payload['sites']),241)
            self.assertEqual(sum(site['asd'] for site in payload['sites']),224)
            for site in payload['sites']:
                example=site['example']
                matrix=np.asarray(example['fc'])
                self.assertEqual(matrix.shape,(116,116))
                self.assertTrue(np.isfinite(matrix).all())
                np.testing.assert_allclose(matrix,matrix.T,atol=1e-7)
                original=self.client.get('/api/health-data/subjects/'+example['subject_id']).json()
                self.assertEqual(original['site'],site['id'])
                np.testing.assert_allclose(matrix,original['fc'],atol=1e-7)
                bold=example['bold']
                self.assertEqual(len(bold['frame_indices']),len(bold['values']))
                self.assertLessEqual(len(bold['values']),128)
                self.assertEqual(bold['frame_indices'][0],0)
                self.assertEqual(bold['frame_indices'][-1],example['n_frames']-1)
                full=next(item for item in original['signals']['series'] if item['roi']==bold['roi'])
                expected=np.asarray(full['values'])[bold['frame_indices']]
                np.testing.assert_allclose(bold['values'],expected,atol=1e-7)

    def test_health_data_filters_and_invalid_bounds(self):
        self.assertEqual(self.client.get('/api/health-data?age_min=30&age_max=10').status_code,422)
        self.assertEqual(self.client.get('/api/health-data?age_min=-1').status_code,422)
        self.assertEqual(self.client.get('/api/health-data?site=not-a-site').status_code,422)
        self.assertEqual(self.client.get('/api/health-data/subjects/not-real').status_code,404)
        self.assertEqual(self.client.get('/api/health-data/subjects/bad$id').status_code,400)

    def test_protection_view_reports_actual_mechanisms_and_scope(self):
        from fedbrain.runner import ROOT
        if not (ROOT/'runs'/'federation-demo-s42'/'federation'/'index.json').is_file():
            self.skipTest('Real protection evidence is not available')
        response=self.client.get('/api/data-protection')
        self.assertEqual(response.status_code,200)
        profile=response.json()
        self.assertEqual(profile['run']['id'],'federation-demo-s42')
        self.assertFalse(profile['deployment']['role_switch_is_authorization'])
        self.assertTrue(profile['deployment']['raw_data_centralized_in_demo'])
        self.assertFalse(profile['channels']['raw_data']['uploaded'])
        self.assertFalse(profile['channels']['private_parameters']['uploaded'])
        self.assertEqual(profile['channels']['shared_model']['parameter_count'],4113)
        self.assertTrue(profile['mechanisms']['perturbation']['observed'])
        self.assertFalse(profile['mechanisms']['paillier']['configured'])
        self.assertFalse(profile['mechanisms']['paillier']['observed'])
        self.assertFalse(profile['mechanisms']['differential_privacy']['certified'])
        self.assertIsNone(profile['mechanisms']['differential_privacy']['epsilon'])
        self.assertFalse(profile['channels']['alignment_statistics']['protected_by_paillier'])
        self.assertFalse(profile['channels']['alignment_statistics']['protected_by_noise'])
        self.assertTrue(profile['evidence']['private_preserved'])
        he=self.client.get('/api/data-protection?run_id=encrypted-smoke-2048').json()
        self.assertTrue(he['mechanisms']['paillier']['configured'])
        self.assertTrue(he['mechanisms']['paillier']['observed'])
        self.assertEqual(he['mechanisms']['paillier']['key_bits'],2048)
        self.assertTrue(he['mechanisms']['paillier']['keys_same_process'])
        self.assertIsNone(he['evidence']['private_preserved'])

    def test_transfer_view_uses_uploaded_and_aggregate_model_coordinates_only(self):
        from fedbrain.runner import ROOT
        if not (ROOT/'runs'/'federation-demo-s42'/'federation'/'index.json').is_file():
            self.skipTest('Actual transfer trace is not available')
        profile=self.client.get('/api/data-protection').json()
        transfer=profile['transfer']
        self.assertTrue(transfer['available'])
        self.assertEqual(transfer['round'],8)
        self.assertEqual(transfer['sketch_size'],64)
        self.assertEqual(len(transfer['sites']),6)
        uploaded=np.asarray([site['uploaded_shared_sketch'] for site in transfer['sites']])
        aggregate=np.asarray(transfer['aggregate_shared_sketch'])
        self.assertEqual(uploaded.shape,(6,64))
        self.assertEqual(aggregate.shape,(64,))
        self.assertTrue(np.isfinite(uploaded).all())
        np.testing.assert_allclose(uploaded.mean(axis=0),aggregate,atol=1e-7)
        for site in transfer['sites']:
            self.assertNotIn('fc',site)
            self.assertNotIn('bold',site)
            self.assertNotIn('private_parameters',site)
        legacy=self.client.get('/api/data-protection?run_id=encrypted-smoke-2048').json()['transfer']
        self.assertFalse(legacy['available'])
        self.assertIsNone(legacy['aggregate_shared_sketch'])

    def test_protection_view_rejects_unknown_and_bad_run(self):
        self.assertEqual(self.client.get('/api/data-protection?run_id=no-such-run').status_code,404)
        self.assertEqual(self.client.get('/api/data-protection?run_id=bad$id').status_code,400)

    def test_unknown_run_and_invalid_identifier(self):
        self.assertEqual(self.client.get('/api/runs/no-such-run').status_code,404)
        self.assertEqual(self.client.get('/api/runs/bad$identifier').status_code,400)

    def test_training_rejects_invalid_input_before_launch(self):
        self.assertEqual(self.client.post('/api/train',json={'rounds':0}).status_code,422)
        self.assertEqual(self.client.post('/api/train',json={'data_dir':'private'}).status_code,422)


if __name__ == '__main__':
    unittest.main()
