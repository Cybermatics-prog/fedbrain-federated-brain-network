# SPDX-License-Identifier: MIT
"""Offline algebraic telemetry tests, with no optimization or training runs."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from fedbrain.federation import FederationRecorder
from fedbrain.model import FedBrainModel
from fedbrain.privacy import average_shared_states


class FederationRecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / 'unit-run'
        self.sites = ['A', 'B', 'C']
        torch.manual_seed(42)
        self.models = {site: FedBrainModel(5, 8) for site in self.sites}
        shared = self.models['A'].shared_state_dict()
        for model in self.models.values():
            model.load_shared_state_dict(shared)
        rng = np.random.default_rng(42)
        self.data = {site: (rng.normal(size=(150 + index, 5)).astype(np.float32),
                            np.arange(150 + index) % 2) for index, site in enumerate(self.sites)}
        self.config = {'rounds': 2, 'hidden_dim': 8, 'noise_scale': .01, 'clip_norm': 10.}

    def recorder(self):
        return FederationRecorder(self.directory, models=self.models, train_data=self.data,
                                  config=self.config, train_subject_counts={s: 10 for s in self.sites},
                                  feature_indices=np.arange(5), site_order=self.sites)

    def read_index(self):
        return json.loads((self.directory / 'federation' / 'index.json').read_text(encoding='utf-8'))

    def read_frame(self, round_index=1):
        return json.loads((self.directory / 'federation' / f'round-{round_index:04d}.json').read_text(encoding='utf-8'))

    @staticmethod
    def states(models):
        return {s: {k: v.detach().clone() for k, v in m.state_dict().items()} for s, m in models.items()}

    def assert_states_equal(self, expected):
        for site, state in expected.items():
            for key, value in state.items():
                torch.testing.assert_close(self.models[site].state_dict()[key], value, rtol=0, atol=0)

    def test_no_rng_input_parameter_gradient_or_mixed_training_flag_mutation(self):
        self.models['A'].train()
        self.models['A'].encoder[0].training = False
        self.models['B'].eval()
        for model in self.models.values():
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
        before = self.states(self.models)
        flags = {s: [m.training for m in model.modules()] for s, model in self.models.items()}
        arrays = {s: (x.copy(), y.copy()) for s, (x, y) in self.data.items()}
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        torch_rng = torch.random.get_rng_state().clone()
        recorder = self.recorder()
        recorder.start_round(1, self.models)
        for site in self.sites:
            recorder.client_started(site)
            recorder.client_uploaded(site, self.models[site], self.models[site].shared_state_dict())
        shared = average_shared_states([self.models[s].shared_state_dict() for s in self.sites])
        recorder.aggregated(shared)
        # Incoming shared weights are already equal, so no external write needed.
        recorder.broadcast(self.models)
        recorder.finish()
        self.assert_states_equal(before)
        self.assertEqual(random.getstate(), python_rng)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], numpy_rng[0])
        np.testing.assert_array_equal(after_numpy[1], numpy_rng[1])
        self.assertEqual(after_numpy[2:], numpy_rng[2:])
        torch.testing.assert_close(torch.random.get_rng_state(), torch_rng, rtol=0, atol=0)
        for site, model in self.models.items():
            self.assertEqual([m.training for m in model.modules()], flags[site])
            for parameter in model.parameters():
                torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
            np.testing.assert_array_equal(self.data[site][0], arrays[site][0])
            np.testing.assert_array_equal(self.data[site][1], arrays[site][1])

    def test_measured_updates_equal_mean_broadcast_and_private_preservation(self):
        recorder = self.recorder()
        recorder.start_round(1, self.models)
        uploaded = []
        for index, site in enumerate(self.sites, 1):
            recorder.client_started(site)
            # Deterministic algebraic edits stand in for already-completed
            # local updates; this unit test runs no optimizer/training loop.
            with torch.no_grad():
                self.models[site].encoder[0].weight.add_(index * .01)
                self.models[site].global_head.bias.add_(index * .2)
                self.models[site].private_head.bias.add_(.5)
                self.models[site].gate[0].bias.sub_(.25)
            clean = self.models[site].shared_state_dict()
            noisy = {key: value + .01 for key, value in clean.items()}
            uploaded.append(noisy)
            recorder.client_uploaded(site, self.models[site], noisy, loss=.6, alignment_loss=.1)
        aggregate = average_shared_states(uploaded)
        recorder.aggregated(aggregate)
        for model in self.models.values():
            model.load_shared_state_dict(aggregate)
        recorder.broadcast(self.models)
        recorder.finish()
        frame, index = self.read_frame(), self.read_index()
        self.assertEqual(frame['status'], 'completed')
        self.assertEqual(index['completed_rounds'], 1)
        self.assertEqual(index['current_phase'], 'ready')
        self.assertTrue(frame['after']['private_preserved'])
        self.assertEqual(frame['after']['shared_consensus_max_error'], 0.)
        self.assertLess(frame['aggregation']['equal_mean_max_error'], 1e-6)
        self.assertGreater(frame['aggregation']['server_delta_l2'], 0.)
        self.assertGreater(frame['aggregation']['mean_local_distance_to_aggregate_l2'], 0.)
        for site in self.sites:
            client = frame['clients'][site]
            self.assertGreater(client['update_l2'], 0.)
            self.assertGreater(client['upload_perturbation_l2'], 0.)
            self.assertEqual(client['upload_bytes'], index['metadata']['shared_parameter_count'] * 4)
            self.assertEqual(client['private_hash'], frame['after']['sites'][site]['private_hash'])
            self.assertNotEqual(frame['before']['sites'][site]['private_hash'], client['private_hash'])
            self.assertEqual(len(client['shared_sketch']), 64)
        events = index['events']
        self.assertEqual([e['sequence'] for e in events], list(range(1, len(events) + 1)))
        self.assertTrue(all(a['monotonic_ns'] <= b['monotonic_ns'] for a, b in zip(events, events[1:])))
        self.assertEqual(sum(e['event'] == 'client_uploaded' for e in events), 3)
        self.assertTrue(any(e['event'] == 'broadcast_verified' for e in events))
        # Strict JSON serialization rejects NaN/Inf rather than emitting them.
        json.dumps(frame, allow_nan=False)
        json.dumps(index, allow_nan=False)

    def test_fixed_train_only_pca_probe_and_no_individual_features_exported(self):
        recorder = self.recorder()
        projection = copy.deepcopy(recorder.index['metadata']['projection'])
        coordinates = copy.deepcopy(recorder.index['metadata']['sketch']['coordinates'])
        recorder.start_round(1, self.models)
        for site in self.sites:
            representation = recorder.frame['before']['sites'][site]['representation']
            self.assertEqual(representation['count'], 128)
            self.assertEqual(set(representation), {'count', 'mean', 'covariance', 'class_centers', 'gate_mean'})
            self.assertEqual(np.asarray(representation['covariance']).shape, (2, 2))
            np.testing.assert_allclose(representation['covariance'], np.asarray(representation['covariance']).T)
            self.assertGreaterEqual(np.linalg.eigvalsh(representation['covariance']).min(), -1e-12)
            self.assertEqual(sum(c['count'] for c in representation['class_centers'].values()), 128)
            indices = np.linspace(0, len(self.data[site][0]) - 1, 128, dtype=int)
            np.testing.assert_array_equal(recorder.probes[site][0], self.data[site][0][indices])
            recorder.client_uploaded(site, self.models[site], self.models[site].shared_state_dict())
        aggregate = average_shared_states([self.models[s].shared_state_dict() for s in self.sites])
        recorder.aggregated(aggregate)
        recorder.broadcast(self.models)
        recorder.start_round(2, self.models)
        self.assertEqual(projection, recorder.index['metadata']['projection'])
        self.assertEqual(coordinates, recorder.index['metadata']['sketch']['coordinates'])
        self.assertEqual(len(coordinates), 64)
        self.assertEqual(projection['fit_scope'], 'initial_training_probe_hidden_only')
        self.assertTrue(projection['fixed'])
        self.assertEqual(np.asarray(projection['components']).shape, (2, 8))

    def test_partial_trace_readable_and_misapplied_broadcast_detected(self):
        recorder = self.recorder()
        self.assertEqual(self.read_index()['current_round'], 0)
        recorder.start_round(1, self.models)
        recorder.client_started('A')
        self.assertEqual(self.read_index()['current_site'], 'A')
        self.assertIsNone(self.read_frame()['aggregation'])
        with self.assertRaises(RuntimeError):
            recorder.aggregated(self.models['A'].shared_state_dict())
        for site in self.sites:
            state = {k: v + .2 for k, v in self.models[site].shared_state_dict().items()}
            recorder.client_uploaded(site, self.models[site], state)
        recorder.aggregated(state)
        with torch.no_grad():
            self.models['A'].private_head.bias.add_(1.)
        recorder.broadcast(self.models)  # Deliberately not actually applied.
        frame = self.read_frame()
        self.assertGreater(frame['after']['shared_consensus_max_error'], 0.)
        self.assertFalse(frame['after']['private_preserved'])
        self.assertFalse(frame['after']['private_preserved_by_site']['A'])
        recorder.finish('failed')
        self.assertEqual(self.read_index()['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
