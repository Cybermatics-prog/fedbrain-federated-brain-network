"""Offline unit tests: synthetic mathematical fixtures, never network results."""
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from fedbrain.data import (N_FEATURES, SITE_ORDER, correlation_matrix,
                           create_subject_splits, features_for_subject, get_fc,
                           read_timeseries, subjects_from_phenotype)
from fedbrain.model import FedBrainModel, domain_alignment_loss, domain_moments
from fedbrain.sparse import SparseFeatureSelector, fit_sparse, prox_two_l1


class ConnectivityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.rng = np.random.default_rng(42)
        self.series = self.rng.normal(size=(40, 116))
        self.path = self.root / 'rois_aal' / 'sample_rois_aal.1D'
        self.path.parent.mkdir()
        np.savetxt(self.path, self.series, header='AAL116')
        self.subject = {'id': 'fixture', 'site': 'NYU', 'label': 1, 'path': str(self.path)}

    def test_pearson_fisher_and_upper_triangle(self):
        actual = correlation_matrix(self.series)
        expected = np.arctanh(np.clip(np.corrcoef(self.series.T), -1 + 1e-7, 1 - 1e-7))
        np.fill_diagonal(expected, 0)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        features = features_for_subject(self.subject)
        self.assertEqual(features.shape, (9, 6670))
        expected_first = correlation_matrix(self.series[:32])[np.triu_indices(116, 1)]
        np.testing.assert_allclose(features[0], expected_first)
        self.assertEqual(get_fc(self.subject).shape, (116, 116))
        np.testing.assert_allclose(actual, actual.T)
        self.assertTrue(np.isfinite(features).all())

    def test_constant_and_perfectly_correlated_rois_are_finite(self):
        self.series[:, 0] = 0
        self.series[:, 2] = self.series[:, 1]
        result = correlation_matrix(self.series)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result[0] == 0).all())
        self.assertGreater(result[1, 2], 8)

    def test_window_stride_subsample_and_cache_validation(self):
        full = features_for_subject(self.subject, stride=2)
        self.assertEqual(full.shape, (5, N_FEATURES))
        subset = features_for_subject(self.subject, stride=2, max_windows=2)
        np.testing.assert_array_equal(subset, full[[0, -1]])
        cache = self.root / 'features' / 'fixture_w32_s2.npy'
        np.save(cache, np.full((5, N_FEATURES), np.nan, dtype=np.float32))
        np.testing.assert_array_equal(features_for_subject(self.subject, stride=2), full)
        self.series[0, 1] = np.nan
        np.savetxt(self.path, self.series)
        with self.assertRaises(ValueError):
            features_for_subject(self.subject, stride=2)

    def test_cache_invalidated_by_valid_source_change(self):
        first = features_for_subject(self.subject)
        self.series[:, 1] = self.series[:, 2]
        np.savetxt(self.path, self.series)
        second = features_for_subject(self.subject)
        self.assertFalse(np.array_equal(first, second))

    def test_invalid_shape_does_not_become_synthetic_data(self):
        np.savetxt(self.path, np.ones((40, 115)))
        with self.assertRaises(ValueError):
            read_timeseries(self.path)


class CohortAndSplitTests(unittest.TestCase):
    def test_quality_filter_and_deterministic_manifest_before_cap(self):
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=['SITE_ID', 'SUB_ID', 'FILE_ID', 'DX_GROUP', 'AGE_AT_SCAN', 'qc_rater_1'])
        writer.writeheader()
        for sid, diagnosis, quality in [('2', 2, 'OK'), ('1', 1, 'OK'), ('3', 1, 'FAIL')]:
            writer.writerow({'SITE_ID': 'NYU', 'SUB_ID': sid, 'FILE_ID': 'NYU_' + sid, 'DX_GROUP': diagnosis,
                             'AGE_AT_SCAN': 10, 'qc_rater_1': quality})
        writer.writerow({'SITE_ID': 'UM_1', 'SUB_ID': '4', 'FILE_ID': 'no_filename', 'DX_GROUP': 1})
        subjects, audit = subjects_from_phenotype(buffer.getvalue(), self._testMethodName, max_per_site=1)
        self.assertEqual([s['id'] for s in subjects], ['1'])
        self.assertEqual(subjects[0]['label'], 1)
        self.assertEqual(audit['available_quality_filtered']['NYU'], 2)
        self.assertEqual(audit['excluded']['quality_fail'], 1)
        self.assertEqual(audit['excluded']['missing_derivative'], 1)
        self.assertFalse(subjects[0]['quality']['all_raters_ok'])

    @staticmethod
    def cohort():
        return [{'id': f'{site}-{index:02d}', 'site': site, 'label': index % 2, 'path': 'not_read'}
                for site in SITE_ORDER for index in range(20)]

    def test_subject_holdout_is_disjoint_stratified_and_order_independent(self):
        subjects = self.cohort()
        first = create_subject_splits(subjects)
        second = create_subject_splits(list(reversed(subjects)))
        for key in ('train', 'test'):
            self.assertEqual([s['id'] for s in first[key]], [s['id'] for s in second[key]])
        train_ids = {s['id'] for s in first['train']}
        test_ids = {s['id'] for s in first['test']}
        self.assertFalse(train_ids & test_ids)
        self.assertEqual(len(train_ids | test_ids), len(subjects))
        # Augmented windows inherit a single subject assignment: no same-subject
        # time segment can appear in both the train and test sets.
        train_windows = {(s['id'], t) for s in first['train'] for t in range(9)}
        test_windows = {(s['id'], t) for s in first['test'] for t in range(9)}
        self.assertFalse(train_windows & test_windows)
        for split in first['by_site'].values():
            self.assertEqual(sum(s['label'] for s in split['test']), 2)

    def test_fivefold_test_sets_partition_subjects(self):
        subjects, test_sets = self.cohort(), []
        for fold in range(5):
            split = create_subject_splits(subjects, fold=fold)
            test_ids = {s['id'] for s in split['test']}
            self.assertFalse(test_ids & {s['id'] for s in split['train']})
            self.assertTrue(all(not test_ids & previous for previous in test_sets))
            test_sets.append(test_ids)
        self.assertEqual(set.union(*test_sets), {s['id'] for s in subjects})
        with self.assertRaises(ValueError):
            create_subject_splits(subjects + [subjects[0]])


class ModelTests(unittest.TestCase):
    def test_experts_gate_logit_mixing_and_gradient(self):
        torch.manual_seed(42)
        model = FedBrainModel(10)
        x = torch.randn(5, 10)
        out = model(x)
        self.assertEqual(out['hidden'].shape, (5, 8))
        self.assertEqual(out['logit'].shape, (5,))
        self.assertTrue(torch.all((out['gate'] > 0) & (out['gate'] < 1)))
        torch.testing.assert_close(out['mixed_logit'], out['gate'] * out['global_logit'] + (1 - out['gate']) * out['private_logit'])
        torch.nn.functional.binary_cross_entropy_with_logits(out['logit'], torch.tensor([0., 1., 0., 1., 0.])).backward()
        for component in (model.encoder, model.global_head, model.private_head, model.gate):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters()))
        with torch.no_grad():
            model.gate[0].weight.zero_()
            model.gate[0].bias.fill_(30)
        torch.testing.assert_close(model(x)['mixed_logit'], model(x)['global_logit'])
        with torch.no_grad():
            model.gate[0].bias.fill_(-30)
        torch.testing.assert_close(model(x)['mixed_logit'], model(x)['private_logit'])

    def test_shared_loading_never_overwrites_private_head_or_gate(self):
        source, target = FedBrainModel(10), FedBrainModel(10)
        private = target.private_state_dict()
        state = source.shared_state_dict()
        self.assertTrue(all(k.startswith(('encoder.', 'global_head.')) for k in state))
        target.load_shared_state_dict(state)
        for key, value in private.items():
            torch.testing.assert_close(target.state_dict()[key], value)
        for key, value in state.items():
            torch.testing.assert_close(target.state_dict()[key], value)
        state[next(iter(state))].zero_()
        self.assertFalse(torch.equal(state[next(iter(state))], source.shared_state_dict()[next(iter(state))]))
        with self.assertRaises(ValueError):
            target.load_shared_state_dict(source.state_dict())

    def test_positive_control_overfits_easiest_separable_task(self):
        # Predeclared pipeline power gate: identical hidden8/expert/gate model,
        # 64 deterministic samples, 150 Adam steps, training accuracy >=.95.
        # This is a unit-test fixture, not evidence about real ABIDE accuracy.
        torch.manual_seed(42)
        inputs = torch.randn(64, 10) * .2
        inputs[:, 0] = torch.cat((torch.linspace(-2., -.25, 32), torch.linspace(.25, 2., 32)))
        labels = (inputs[:, 0] > 0).float()
        self.assertEqual(float(((inputs[:, 0] > 0).float() == labels).float().mean()), 1.)
        model = FedBrainModel(10, hidden_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=.03)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for _ in range(150):
                output = model(inputs)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(output['mixed_logit'], labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                accuracy = ((model(inputs)['mixed_logit'] > 0) == labels.bool()).float().mean().item()
            self.assertGreaterEqual(accuracy, .95)
        finally:
            torch.set_num_threads(previous_threads)

    def test_marginal_and_conditional_alignment_and_missing_class(self):
        hidden = torch.tensor([[0., 0.], [2., 0.], [0., 2.], [2., 2.]], requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        reference = domain_moments(hidden.detach() + 1, labels)
        self.assertFalse(reference['marginal'].requires_grad)
        loss = domain_alignment_loss(hidden, labels, reference)
        torch.testing.assert_close(loss, torch.tensor(3 * np.sqrt(2), dtype=torch.float32))
        loss.backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        torch.testing.assert_close(domain_alignment_loss(hidden, labels, domain_moments(hidden, labels)), torch.tensor(0.))
        absent = domain_moments(torch.tensor([[1., 1.]]), torch.tensor([0]))
        self.assertNotIn(1, absent['conditional'])
        self.assertTrue(torch.isfinite(domain_alignment_loss(hidden, labels, absent)))


class SparseTests(unittest.TestCase):
    def test_proximal_map_matches_scalar_optimization_and_has_true_zeros(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            value, shared = rng.normal(size=2)
            a, b = rng.uniform(.01, 1, size=2)
            actual = float(prox_two_l1(value, shared, a, b))
            objective = lambda x: .5 * (x - value) ** 2 + a * abs(x) + b * abs(x - shared)
            result = minimize_scalar(objective, bounds=(-8, 8), method='bounded', options={'xatol': 1e-12})
            self.assertLessEqual(objective(actual), result.fun + 1e-8)
        self.assertEqual(float(prox_two_l1(.01, 0., .1, .2)), 0.)

    @staticmethod
    def signal():
        rng = np.random.default_rng(42)
        y = np.repeat([0, 1], 60)
        X = np.zeros((120, 16))
        X[:, :3] = (2 * y[:, None] - 1) * 2 + rng.normal(scale=.3, size=(120, 3))
        X[:, 3:6] = rng.normal(size=(120, 3))
        return X, y, np.tile(np.arange(3), 40)

    def test_alternating_sparse_selector_true_zeros_and_explicit_cap(self):
        X, y, segments = self.signal()
        selector = SparseFeatureSelector(l1=.5, coupling=.2, steps=60).fit(X, y, segments)
        self.assertTrue(set((0, 1, 2)).issubset(selector.selected_indices_))
        self.assertTrue((selector.coef_[6:] == 0).all())
        self.assertFalse(selector.diagnostics_['feature_cap_applied'])
        history = selector.diagnostics_['objective_history']
        self.assertTrue(np.all(np.diff(history) <= 1e-7))
        capped = SparseFeatureSelector(l1=.5, coupling=.2, steps=60, max_features=1).fit(X, y, segments)
        self.assertEqual(capped.transform(X).shape, (120, 1))
        self.assertTrue(capped.diagnostics_['feature_cap_applied'])
        self.assertIsNotNone(capped.diagnostics_['approximation'])

    def test_client_union_deterministic_and_never_uses_test_data(self):
        X, y, segments = self.signal()
        clients = {'NYU': {'X': X, 'y': y, 'segments': segments},
                   'UM': {'X': X.copy(), 'y': y.copy(), 'segments': segments}}
        indices, diagnostics = fit_sparse(clients, lambda_aug=.5, coupling=.2, max_iter=20)
        again, _ = fit_sparse(dict(reversed(list(clients.items()))), lambda_aug=.5, coupling=.2, max_iter=20)
        np.testing.assert_array_equal(indices, again)
        self.assertEqual(diagnostics['merge'], 'union_of_per_site_nonzero_shared_coefficients')
        self.assertFalse(diagnostics['feature_cap_applied'])
        capped, capped_diagnostics = fit_sparse(clients, lambda_aug=.5, coupling=.2, max_iter=20, max_features=1)
        self.assertEqual(len(capped), 1)
        self.assertTrue(capped_diagnostics['feature_cap_applied'])


if __name__ == '__main__':
    unittest.main()
