# SPDX-License-Identifier: MIT
"""Numerical equivalence of grouped BLAS to the original Eq.11 implementation."""
import unittest

import numpy as np

from fedbrain.sparse import (SparseFeatureSelector, _GroupedLogisticProblem,
                             _shared_median, prox_two_l1, soft_threshold)


def reference_gradient(X, y, encoded, weights, biases):
    """Original pre-acceleration implementation, retained only as a test oracle."""
    scores = np.einsum('ij,ij->i', X, weights[encoded]) + biases[encoded]
    probabilities = np.empty_like(scores)
    positive = scores >= 0
    probabilities[positive] = 1 / (1 + np.exp(-scores[positive]))
    exponential = np.exp(scores[~positive])
    probabilities[~positive] = exponential / (1 + exponential)
    residual = probabilities - y
    grad = np.zeros_like(weights)
    np.add.at(grad, encoded, X * residual[:, None])
    return grad, np.bincount(encoded, weights=residual, minlength=len(weights))


def reference_loss(X, y, encoded, weights, biases):
    scores = np.einsum('ij,ij->i', X, weights[encoded]) + biases[encoded]
    return float(np.logaddexp(0, scores).sum() - np.dot(y, scores))


def reference_fit(X, y, segments, steps=40, l1=.01, coupling=1., learning_rate=.1, tolerance=1e-6):
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    segment_ids, encoded = np.unique(segments, return_inverse=True)
    weights, biases = np.zeros((len(segment_ids), X.shape[1])), np.zeros(len(segment_ids))
    bound = .25 * np.bincount(encoded, weights=np.sum(X * X, axis=1) + 1, minlength=len(segment_ids))
    step = np.minimum(learning_rate, 1 / np.maximum(bound, 1e-12))[:, None]
    for _ in range(min(8, steps)):
        grad, bias_grad = reference_gradient(X, y, encoded, weights, biases)
        weights = soft_threshold(weights - step * grad, step * l1)
        biases -= step[:, 0] * bias_grad
    shared = _shared_median(weights)

    def objective():
        return reference_loss(X, y, encoded, weights, biases) + l1 * np.abs(weights).sum() + coupling * np.abs(weights - shared).sum()

    history = [objective()]
    for _ in range(steps):
        previous, previous_biases = weights.copy(), biases.copy()
        shared = _shared_median(weights)
        grad, bias_grad = reference_gradient(X, y, encoded, weights, biases)
        weights = prox_two_l1(weights - step * grad, shared[None], step * l1, step * coupling)
        biases -= step[:, 0] * bias_grad
        shared = _shared_median(weights)
        history.append(objective())
        if max(np.max(np.abs(weights - previous)), np.max(np.abs(biases - previous_biases))) <= tolerance:
            break
    return shared, weights, biases, np.asarray(history)


class SparseGroupedEquivalenceTests(unittest.TestCase):
    @staticmethod
    def fixture():
        rng = np.random.default_rng(4096)
        X = rng.normal(size=(128, 25))
        X[:, -2:] = 0
        labels = (X[:, 0] + .3 * X[:, 1] > 0).astype(np.float64)
        # Deliberately interleaved groups, as in the actual subject-major input.
        encoded = np.tile(np.arange(8), 16)
        weights, biases = rng.normal(size=(8, 25)), rng.normal(size=8)
        return X, labels, encoded, weights, biases

    def test_grouped_gradient_and_objective_match_old_reductions(self):
        X, labels, encoded, weights, biases = self.fixture()
        problem = _GroupedLogisticProblem(X, labels, encoded, 8)
        grad, bias_grad = problem.gradient(weights, biases)
        reference, reference_bias = reference_gradient(X, labels, encoded, weights, biases)
        np.testing.assert_allclose(grad, reference, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(bias_grad, reference_bias, atol=1e-12, rtol=1e-12)
        self.assertAlmostEqual(problem.loss(weights, biases), reference_loss(X, labels, encoded, weights, biases), delta=1e-10)

    def test_summed_logistic_not_mean_and_finite_extreme_scores(self):
        X, labels, encoded, weights, biases = self.fixture()
        once = _GroupedLogisticProblem(X, labels, encoded, 8)
        twice = _GroupedLogisticProblem(np.concatenate((X, X)), np.tile(labels, 2), np.tile(encoded, 2), 8)
        first_grad, first_bias = once.gradient(weights, biases)
        second_grad, second_bias = twice.gradient(weights, biases)
        np.testing.assert_allclose(second_grad, 2 * first_grad, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(second_bias, 2 * first_bias, atol=1e-12, rtol=1e-12)
        self.assertAlmostEqual(twice.loss(weights, biases), 2 * once.loss(weights, biases), delta=1e-10)
        huge_weights, huge_bias = weights * 1e4, biases * 1e4
        self.assertTrue(np.isfinite(once.loss(huge_weights, huge_bias)))
        grad, bias = once.gradient(huge_weights, huge_bias)
        self.assertTrue(np.isfinite(grad).all() and np.isfinite(bias).all())

    def test_exact_proximal_update_unchanged(self):
        X, labels, encoded, weights, biases = self.fixture()
        problem = _GroupedLogisticProblem(X, labels, encoded, 8)
        actual, _ = problem.gradient(weights, biases)
        expected, _ = reference_gradient(X, labels, encoded, weights, biases)
        shared, step = _shared_median(weights), 1e-3
        actual_update = prox_two_l1(weights - step * actual, shared[None], step * .01, step)
        expected_update = prox_two_l1(weights - step * expected, shared[None], step * .01, step)
        np.testing.assert_allclose(actual_update, expected_update, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(actual_update == 0, expected_update == 0)

    def test_default_forty_step_coefficients_mask_and_history_preserved(self):
        X, labels, encoded, _, _ = self.fixture()
        coef, weights, biases, history = reference_fit(X, labels, encoded)
        fitted = SparseFeatureSelector().fit(X, labels, encoded)
        np.testing.assert_allclose(fitted.coef_, coef, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(fitted.segment_coef_, weights, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(fitted.segment_intercepts_, biases, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(fitted.diagnostics_['objective_history'], history, atol=1e-10, rtol=1e-10)
        np.testing.assert_array_equal(fitted.selected_indices_, np.flatnonzero(coef != 0))
        self.assertEqual(fitted.steps, 40)
        self.assertIsNone(fitted.max_features)

    def test_longer_optimization_reports_real_changes_not_certified_gap(self):
        X, labels, encoded, _, _ = self.fixture()
        short = SparseFeatureSelector(steps=40).fit(X, labels, encoded)
        longer = SparseFeatureSelector(steps=400).fit(X, labels, encoded)
        self.assertLessEqual(longer.diagnostics_['objective_final'], short.diagnostics_['objective_final'] + 1e-10)
        self.assertTrue(np.all(np.diff(longer.diagnostics_['objective_history']) <= 1e-10))
        self.assertTrue((longer.coef_[-2:] == 0).all())
        self.assertIn('not a certified', longer.diagnostics_['convergence_note'])
        self.assertGreaterEqual(longer.diagnostics_['relative_objective_change_last'], 0)
        self.assertEqual(longer.diagnostics_['gradient_backend'], 'grouped_float64_BLAS_summed_logistic')


if __name__ == '__main__':
    unittest.main()
