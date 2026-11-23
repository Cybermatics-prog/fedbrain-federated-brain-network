# SPDX-License-Identifier: MIT
"""Recorded-JSON fixtures for protocol facts; no model/raw/crypto execution."""
import json
from pathlib import Path
import tempfile
import unittest

from fedbrain.protection import describe_protection


class ProtectionFactsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / 'recorded-run'
        self.directory.mkdir()

    def result(self, noise=.01, he=False, rounds=1, method='fedbrain', weight=.1, status='completed'):
        value = {'id': 'recorded-run', 'status': status, 'round': rounds,
                 'config': {'method': method, 'rounds': 8, 'noise_scale': noise,
                            'clip_norm': 10., 'homomorphic': he, 'paillier_bits': 2048,
                            'alignment_weight': weight}}
        (self.directory / 'result.json').write_text(json.dumps(value), encoding='utf-8')
        return value

    def trace(self, preserved=True):
        folder = self.directory / 'federation'
        folder.mkdir(exist_ok=True)
        index = {'completed_rounds': 1, 'rounds': [{'round': 1, 'status': 'completed', 'path': '../../not-allowed.json'}],
                 'metadata': {'shared_parameter_count': 4113,
                              'shared_parameter_names': ['encoder.0.weight', 'encoder.0.bias', 'global_head.weight', 'global_head.bias'],
                              'private_parameter_count': 18},
                 'events': [{'event': 'client_uploaded', 'round': 1, 'site': site} for site in ('A', 'B')]}
        frame = {'round': 1, 'clients': {'A': {'upload_bytes': 16452}, 'B': {'upload_bytes': 16452}},
                 'aggregation': {'equal_mean_max_error': 1e-8},
                 'after': {'private_preserved': preserved, 'shared_consensus_max_error': 0.}}
        (folder / 'index.json').write_text(json.dumps(index), encoding='utf-8')
        (folder / 'round-0001.json').write_text(json.dumps(frame), encoding='utf-8')
        return index, frame

    def test_unavailable_is_not_security_verification_and_demo_boundaries_explicit(self):
        data = describe_protection(None)
        self.assertFalse(data['available'])
        self.assertTrue(data['deployment']['raw_data_centralized_in_demo'])
        self.assertTrue(data['deployment']['individual_data_api_available'])
        self.assertFalse(data['deployment']['role_switch_is_authorization'])
        self.assertFalse(data['mechanisms']['perturbation']['observed'])
        self.assertFalse(data['mechanisms']['paillier']['observed'])
        self.assertIsNone(data['evidence']['private_preserved'])
        self.assertIsNone(data['channels']['shared_model']['parameter_count'])
        self.assertIsNone(data['mechanisms']['differential_privacy']['epsilon'])
        self.assertFalse(data['mechanisms']['differential_privacy']['certified'])

    def test_noise_and_he_combinations_distinguish_configured_applied_and_proven(self):
        for noise in (0., .01):
            for he in (False, True):
                with self.subTest(noise=noise, he=he):
                    self.result(noise=noise, he=he)
                    data = describe_protection(self.directory)
                    self.assertEqual(data['mechanisms']['perturbation']['configured'], noise > 0)
                    self.assertEqual(data['mechanisms']['perturbation']['observed'], noise > 0)
                    self.assertEqual(data['mechanisms']['paillier']['configured'], he)
                    self.assertEqual(data['mechanisms']['paillier']['observed'], he)
                    self.assertEqual(data['mechanisms']['paillier']['key_bits'], 2048 if he else None)
                    self.assertTrue(data['mechanisms']['paillier']['keys_same_process'])
                    self.assertFalse(data['mechanisms']['perturbation']['formal_dp'])
                    self.assertFalse(data['evidence']['trace_available'])
                    self.assertIsNone(data['evidence']['private_preserved'])
                    self.assertIsNone(data['channels']['private_parameters']['parameter_count_per_site'])
                    self.assertIn('not_cryptographic_proof', data['mechanisms']['paillier']['observation_basis']) if he else None
                    stats = data['channels']['alignment_statistics']
                    self.assertFalse(stats['protected_by_paillier'])
                    self.assertFalse(stats['protected_by_noise'])

    def test_configured_r0_not_observed_preparing_or_stopped(self):
        for status in ('preparing', 'stopped', 'failed'):
            self.result(noise=.01, he=True, rounds=0, status=status)
            data = describe_protection(self.directory)
            self.assertTrue(data['mechanisms']['perturbation']['configured'])
            self.assertTrue(data['mechanisms']['paillier']['configured'])
            self.assertFalse(data['mechanisms']['perturbation']['observed'])
            self.assertFalse(data['mechanisms']['paillier']['observed'])
            self.assertEqual(data['run']['rounds_completed'], 0)
            self.assertEqual(data['evidence']['client_uploads'], 0)

    def test_method_and_alignment_weight_follow_actual_runner_guard(self):
        for method, weight, expected in [('fedbrain', .1, True), ('no_personalization', .1, True),
                                         ('fedavg', .1, False), ('no_alignment', .1, False),
                                         ('fedbrain', 0., False)]:
            self.result(method=method, weight=weight)
            data = describe_protection(self.directory)
            channel = data['channels']['alignment_statistics']
            self.assertEqual(channel['uploaded'], expected)
            self.assertEqual(channel['items'], ['marginal_mean', 'class_means', 'class_counts'] if expected else [])
            self.assertFalse(data['channels']['raw_data']['uploaded'])
            self.assertFalse(data['channels']['private_parameters']['uploaded'])
            self.assertTrue(data['channels']['shared_model']['uploaded'])

    def test_actual_trace_counts_zero_error_and_private_failure_are_not_coerced(self):
        self.result()
        self.trace(preserved=False)
        data = describe_protection(self.directory)
        self.assertTrue(data['evidence']['trace_available'])
        self.assertEqual(data['evidence']['recorded_rounds'], 1)
        self.assertEqual(data['evidence']['client_uploads'], 2)
        self.assertIs(data['evidence']['private_preserved'], False)
        self.assertEqual(data['evidence']['shared_consensus_max_error'], 0.)
        self.assertEqual(data['evidence']['equal_mean_max_error'], 1e-8)
        self.assertEqual(data['channels']['shared_model']['parameter_count'], 4113)
        self.assertEqual(data['channels']['shared_model']['raw_tensor_bytes'], 16452)
        self.assertEqual(data['channels']['private_parameters']['parameter_count_per_site'], 18)
        self.assertEqual(data['evidence']['paths']['latest_round'], 'federation/round-0001.json')
        json.dumps(data, allow_nan=False)

    def test_actual_run_cohort_scope_window_counts_and_r0_inventory_not_delivery(self):
        for total, train, he in ((465, 292, False), (120, 72, True)):
            with self.subTest(total=total, train=train, he=he):
                value = self.result(he=he, rounds=0, status='preparing')
                value['data_subjects'] = total
                value['split_counts'] = {'train': train, 'val': 24, 'test': 24}
                (self.directory / 'result.json').write_text(json.dumps(value), encoding='utf-8')
                data = describe_protection(self.directory)
                self.assertEqual(data['run']['data_subjects'], total)
                self.assertEqual(data['run']['train_subjects'], train)
                self.assertEqual(data['run']['cohort_scope'], 'actual_selected_run_not_current_health_data_filter')
                shared = data['channels']['shared_model']
                self.assertTrue(shared['uploaded'])  # Protocol inventory, not delivery.
                self.assertEqual(shared['scope'], 'training_protocol')
                self.assertIn('R0', shared['note'])
                self.assertIn('packet capture', shared['note'])
                self.assertEqual(data['evidence']['client_uploads'], 0)
                self.assertFalse(data['mechanisms']['paillier']['observed'])
                stats = data['channels']['alignment_statistics']
                self.assertEqual(stats['class_counts_unit'], 'training_window_samples_not_patients')
                self.assertFalse(stats['protected_by_noise'])
                self.assertFalse(stats['protected_by_paillier'])
        self.result()
        unknown = describe_protection(self.directory)
        self.assertIsNone(unknown['run']['data_subjects'])
        self.assertIsNone(unknown['run']['train_subjects'])

    def test_transfer_uses_exact_recorded_upload_clean_delta_and_aggregate_coordinates(self):
        self.result()
        index, frame = self.trace(preserved=False)
        index['site_order'] = ['A', 'B']
        index['metadata']['sites'] = {'A': {'train_subject_count': 109}, 'B': {'train_subject_count': 56}}
        index['metadata']['sketch'] = {'coordinates': [{'flat_index': i, 'parameter': 'encoder.0.weight',
                                                       'parameter_index': [0, i]} for i in range(64)]}
        for offset, site in enumerate(('A', 'B'), 1):
            frame['clients'][site].update(shared_sketch=[i * .01 + offset * .1 for i in range(64)],
                                          clean_shared_sketch=[i * .01 for i in range(64)],
                                          update_sketch=[offset * .001 for _ in range(64)])
        frame['aggregation']['shared_sketch'] = [i * .01 + .15 for i in range(64)]
        frame['after']['private_preserved_by_site'] = {'A': True, 'B': False}
        folder = self.directory / 'federation'
        (folder / 'index.json').write_text(json.dumps(index), encoding='utf-8')
        (folder / 'round-0001.json').write_text(json.dumps(frame), encoding='utf-8')
        before_bytes = (folder / 'round-0001.json').read_bytes()
        data = describe_protection(self.directory)
        transfer = data['transfer']
        self.assertTrue(transfer['available'])
        self.assertEqual(transfer['round'], 1)
        self.assertEqual(transfer['sketch_size'], 64)
        self.assertEqual(transfer['aggregate_shared_sketch'], frame['aggregation']['shared_sketch'])
        self.assertEqual(transfer['coordinates'], index['metadata']['sketch']['coordinates'])
        for site in transfer['sites']:
            original = frame['clients'][site['id']]
            self.assertEqual(site['uploaded_shared_sketch'], original['shared_sketch'])
            self.assertEqual(site['clean_shared_sketch'], original['clean_shared_sketch'])
            self.assertEqual(site['local_update_sketch'], original['update_sketch'])
            self.assertEqual(site['private_preserved'], frame['after']['private_preserved_by_site'][site['id']])
            self.assertEqual(site['train_subjects'], index['metadata']['sites'][site['id']]['train_subject_count'])
        statistics = transfer['alignment_statistics']
        self.assertFalse(statistics['values_recorded'])
        self.assertIsNone(statistics['marginal_mean'])
        self.assertIsNone(statistics['class_means'])
        self.assertIsNone(statistics['class_counts'])
        self.assertFalse(statistics['protected_by_noise'])
        self.assertFalse(statistics['protected_by_paillier'])
        self.assertEqual((folder / 'round-0001.json').read_bytes(), before_bytes)
        json.dumps(data, allow_nan=False)

    def test_transfer_unknown_not_pca_or_incoming_model_substitution(self):
        self.result(he=True)
        legacy = describe_protection(self.directory)['transfer']
        self.assertFalse(legacy['available'])
        self.assertEqual(legacy['sites'], [])
        self.assertIsNone(legacy['aggregate_shared_sketch'])
        _, frame = self.trace()
        frame['before'] = {'shared_sketch': [99.] * 64, 'sites': {'A': {'representation': {'mean': [1., 2.]}}}}
        frame['clients']['A']['shared_sketch'] = [1.] * 63
        frame['clients']['B']['shared_sketch'] = [float('nan')] * 64
        frame['aggregation']['shared_sketch'] = [0.] * 63
        path = self.directory / 'federation' / 'round-0001.json'
        path.write_text(json.dumps(frame), encoding='utf-8')
        transfer = describe_protection(self.directory)['transfer']
        self.assertFalse(transfer['available'])
        self.assertIsNone(transfer['aggregate_shared_sketch'])
        self.assertTrue(all(site['uploaded_shared_sketch'] is None for site in transfer['sites']))
        json.dumps(transfer, allow_nan=False)

    def test_no_checkpoint_or_raw_data_read_and_missing_frame_remains_unknown(self):
        value = self.result(he=True)
        value['config']['data_dir'] = 'not-a-real-raw-data-directory'
        value['config']['hidden_dim'] = 9999
        value['config']['max_features'] = 12345
        (self.directory / 'result.json').write_text(json.dumps(value), encoding='utf-8')
        # Invalid binary data would explode if a checkpoint loader were used.
        (self.directory / 'checkpoint.pt').write_bytes(b'not-a-checkpoint\xff\xfe')
        folder = self.directory / 'federation'
        folder.mkdir()
        (folder / 'index.json').write_text(json.dumps({'completed_rounds': 1, 'rounds': [{'round': 1}]}), encoding='utf-8')
        data = describe_protection(self.directory)
        self.assertFalse(data['evidence']['trace_available'])
        self.assertEqual(data['evidence']['recorded_rounds'], 0)
        self.assertIsNone(data['channels']['shared_model']['parameter_count'])
        self.assertIsNone(data['channels']['shared_model']['raw_tensor_bytes'])
        self.assertEqual(data['channels']['shared_model']['names'], [])
        self.assertIsNone(data['evidence']['shared_consensus_max_error'])
        self.assertIsNone(data['evidence']['equal_mean_max_error'])
        self.assertTrue(data['mechanisms']['paillier']['observed'])
        self.assertFalse(data['evidence']['mechanism_observation_is_cryptographic_proof'])


if __name__ == '__main__':
    unittest.main()
