"""Train-only shared/segment sparse logistic selection from paper Eq.11.

For each site we minimize summed logistic losses + l1*sum|theta_t| +
coupling*sum|theta_t-theta|. Alternation uses a coordinate-wise median shared
update (the exact L1 minimizer) and proximal-gradient segment updates. Unlike
an L1 term added to Adam, the exact proximal map produces literal zeroes.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np


def soft_threshold(values, threshold):
    values = np.asarray(values)
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0)


def prox_two_l1(values, shared, l1_step, coupling_step):
    """Exact prox of a|x|+b|x-c|, including both nonsmooth breakpoints.

    The optimum is either 0, c, or one of the four possible stationary roots;
    comparing their actual objective gives the exact scalar solution in batch.
    """
    values, shared = np.broadcast_arrays(np.asarray(values), np.asarray(shared))
    a, b = np.asarray(l1_step), np.asarray(coupling_step)
    best = np.zeros_like(values)
    best_objective = .5 * values ** 2 + b * np.abs(shared)
    for candidate in (shared, values - a - b, values - a + b,
                      values + a - b, values + a + b):
        objective = .5 * (candidate - values) ** 2 + a * np.abs(candidate) + b * np.abs(candidate - shared)
        use = objective < best_objective
        best = np.where(use, candidate, best)
        best_objective = np.minimum(best_objective, objective)
    return best


def _shared_median(segment_weights):
    # Even-sample median minimizers form an interval. Choosing its closest point
    # to zero preserves true zeros without changing the optimization objective.
    count = len(segment_weights)
    lower_i, upper_i = (count - 1) // 2, count // 2
    partitioned = np.partition(segment_weights, (lower_i, upper_i), axis=0)
    return np.clip(np.zeros(segment_weights.shape[1]), partitioned[lower_i], partitioned[upper_i])


class _GroupedLogisticProblem:
    """Summed logistic loss/gradient using one contiguous BLAS block/segment.

    Grouping is performed once. Per-iteration memory is O(T*d + n), rather
    than materializing both weights[encoded] and X*residual with shape[n,d].
    Input order within a group is preserved; only floating reduction order
    differs from the original einsum/add.at implementation.
    """

    def __init__(self, X, y, encoded, n_segments):
        self.dimension = X.shape[1]
        self.groups = [
            (np.ascontiguousarray(X[encoded == segment]),
             np.ascontiguousarray(y[encoded == segment]))
            for segment in range(n_segments)
        ]

    def gradient(self, weights, biases):
        grad, bias_grad = np.empty_like(weights), np.empty_like(biases)
        for segment, (matrix, labels) in enumerate(self.groups):
            scores = matrix @ weights[segment] + biases[segment]
            probabilities = np.empty_like(scores)
            positive = scores >= 0
            probabilities[positive] = 1 / (1 + np.exp(-scores[positive]))
            exponential = np.exp(scores[~positive])
            probabilities[~positive] = exponential / (1 + exponential)
            residual = probabilities - labels
            grad[segment] = matrix.T @ residual
            bias_grad[segment] = residual.sum(dtype=np.float64)
        return grad, bias_grad

    def loss(self, weights, biases):
        logistic = 0.
        for segment, (matrix, labels) in enumerate(self.groups):
            scores = matrix @ weights[segment] + biases[segment]
            logistic += float(np.logaddexp(0, scores).sum() - np.dot(labels, scores))
        return logistic


class SparseFeatureSelector:
    """Fit one site's *training* windows only, then expose a reusable mask.

    ``segments`` is one integer/time-position ID per window. With None the
    caller explicitly fits a single-segment/static variant. There is no hidden
    row budget and no default feature cap. Capping is an explicit approximation
    ranked by absolute shared coefficients; it never invents nonzero features.
    Standardization, if desired, must already be fitted on training data only.
    """

    def __init__(self, l1=.01, coupling=1., steps=40, max_features=None,
                 seed=42, learning_rate=.1, tolerance=1e-6):
        if l1 < 0 or coupling < 0 or steps < 1 or learning_rate <= 0:
            raise ValueError("l1/coupling >=0, steps >=1, learning_rate >0 required")
        if max_features is not None and (not isinstance(max_features, int) or max_features < 1):
            raise ValueError("max_features must be positive or None")
        self.l1, self.coupling = float(l1), float(coupling)
        self.steps, self.max_features, self.seed = int(steps), max_features, int(seed)
        self.learning_rate, self.tolerance = float(learning_rate), float(tolerance)

    def fit(self, X, y, segments=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2 or len(X) == 0 or X.shape[1] == 0 or len(y) != len(X):
            raise ValueError("Expected nonempty X[n,d] and matching y[n]")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Non-finite sparse training data")
        if not np.isin(y, (0, 1)).all() or len(np.unique(y)) != 2:
            raise ValueError("Training labels must contain both binary classes 0 and1")
        if segments is None:
            segments = np.zeros(len(X), dtype=np.int64)
        segments = np.asarray(segments).reshape(-1)
        if len(segments) != len(X):
            raise ValueError("segments must have one ID per training window")
        self.segment_ids_, encoded = np.unique(segments, return_inverse=True)
        n_segments, dimension = len(self.segment_ids_), X.shape[1]
        weights, biases = np.zeros((n_segments, dimension)), np.zeros(n_segments)
        # Frobenius bound includes intercept; conservative but guarantees a
        # smooth logistic-gradient step no larger than inverse Lipschitz bound.
        bound = .25 * np.bincount(encoded, weights=np.sum(X * X, axis=1) + 1, minlength=n_segments)
        step = np.minimum(self.learning_rate, 1. / np.maximum(bound, 1e-12))[:, None]

        problem = _GroupedLogisticProblem(X, y, encoded, n_segments)
        gradient = problem.gradient

        # Independent L1 warm start avoids the well-known coordinatewise-zero
        # trap of a nonsmooth |theta_t-theta| consensus block initialized at0.
        warm_steps = min(8, self.steps)
        for _ in range(warm_steps):
            grad, bias_grad = gradient(weights, biases)
            weights = soft_threshold(weights - step * grad, step * self.l1)
            biases -= step[:, 0] * bias_grad
        shared = _shared_median(weights)

        def objective():
            logistic = problem.loss(weights, biases)
            return float(logistic + self.l1 * np.abs(weights).sum() + self.coupling * np.abs(weights - shared).sum())

        history, converged = [objective()], False
        for iteration in range(self.steps):
            previous = weights.copy()
            previous_biases = biases.copy()
            # Exact shared minimization and exact two-L1 segment proximal map.
            shared = _shared_median(weights)
            grad, bias_grad = gradient(weights, biases)
            weights = prox_two_l1(weights - step * grad, shared[None, :],
                                  step * self.l1, step * self.coupling)
            biases -= step[:, 0] * bias_grad
            shared = _shared_median(weights)
            history.append(objective())
            last_weight_change = float(np.max(np.abs(weights - previous)))
            last_bias_change = float(np.max(np.abs(biases - previous_biases)))
            if max(last_weight_change, last_bias_change) <= self.tolerance:
                converged = True
                break
        self.coef_ = shared
        self.segment_coef_, self.segment_intercepts_ = weights, biases
        selected = np.flatnonzero(shared != 0).astype(np.int64)
        uncapped = len(selected)
        if self.max_features is not None and len(selected) > self.max_features:
            order = np.lexsort((selected, -np.abs(shared[selected])))
            selected = np.sort(selected[order[:self.max_features]])
        self.selected_indices_ = selected
        self.n_features_in_ = dimension
        self.diagnostics_ = {
            "objective": "sum logistic + l1*sum|segment| + coupling*sum|shared-segment|",
            "optimizer": "independent-L1 warmstart + exact-median / exact-two-L1-prox alternating PGD",
            "l1": self.l1, "coupling": self.coupling, "training_windows": len(X),
            "segments": n_segments, "single_segment_approximation": segments is not None and n_segments == 1,
            "steps_requested": self.steps, "steps_run": iteration + 1, "warm_start_steps": warm_steps,
            "converged": converged, "objective_history": history,
            "gradient_backend": "grouped_float64_BLAS_summed_logistic",
            "objective_initial": history[0], "objective_final": history[-1],
            "relative_objective_change_last": float(abs(history[-1] - history[-2]) / max(abs(history[-2]), 1e-12)),
            "last_weight_change_max": last_weight_change, "last_bias_change_max": last_bias_change,
            "convergence_tolerance": self.tolerance,
            "step_size_min": float(step.min()), "step_size_max": float(step.max()),
            "convergence_note": "Parameter-change stopping rule, not a certified global optimality gap",
            "uncapped_nonzero": uncapped, "selected_count": len(selected),
            "segment_nonzero": int(np.count_nonzero(weights)),
            "max_features": self.max_features,
            "feature_cap_applied": len(selected) < uncapped,
            "approximation": "top_absolute_shared_coefficient_cap" if len(selected) < uncapped else None,
            "fit_scope": "caller-provided training windows only", "seed": self.seed,
        }
        return self

    def transform(self, X):
        if not hasattr(self, "selected_indices_"):
            raise RuntimeError("Call fit before transform")
        array = np.asarray(X)
        if array.ndim != 2 or array.shape[1] != self.n_features_in_:
            raise ValueError("Input feature dimension differs from fitted selector")
        return array[:, self.selected_indices_]

    def fit_transform(self, X, y, segments=None):
        return self.fit(X, y, segments).transform(X)


def fit_sparse(site_data: Mapping, lambda_aug=.01, coupling=1., max_iter=40,
               learning_rate=.1, max_features=None, seed=42):
    """Fit one selector/site, union nonzero shared coordinates, optionally cap.

    site_data = {site: {'X': train_X, 'y': train_y, 'segments': optional_ids}}.
    Global cap uses the largest absolute coefficient across clients, and is
    reported separately from the uncapped union. No test arrays are accepted or
    used implicitly; the caller owns the prior subject split.
    """
    selectors, dimension, scores = {}, None, None
    for site in sorted(site_data):
        values = site_data[site]
        if not isinstance(values, Mapping) or "X" not in values or "y" not in values:
            raise ValueError("Each site must contain training X and y")
        selector = SparseFeatureSelector(l1=lambda_aug, coupling=coupling, steps=max_iter,
                                         max_features=None, seed=seed, learning_rate=learning_rate)
        selector.fit(values["X"], values["y"], segments=values.get("segments"))
        if dimension is None:
            dimension = selector.n_features_in_
            scores = np.zeros(dimension)
        if selector.n_features_in_ != dimension:
            raise ValueError("All clients must use the same original feature dimension")
        scores = np.maximum(scores, np.abs(selector.coef_))
        selectors[site] = selector
    if not selectors:
        raise ValueError("At least one training site is required")
    selected = np.flatnonzero(scores != 0).astype(np.int64)
    uncapped = len(selected)
    if max_features is not None:
        if not isinstance(max_features, int) or max_features < 1:
            raise ValueError("max_features must be positive or None")
        if len(selected) > max_features:
            order = np.lexsort((selected, -scores[selected]))
            selected = np.sort(selected[order[:max_features]])
    diagnostics = {
        "sites": {site: selector.diagnostics_ for site, selector in selectors.items()},
        "merge": "union_of_per_site_nonzero_shared_coefficients",
        "uncapped_union_count": uncapped, "selected_count": len(selected),
        "max_features": max_features, "feature_cap_applied": len(selected) < uncapped,
        "approximation": "top_max_absolute_client_coefficient_cap" if len(selected) < uncapped else None,
        "fit_scope": "training windows only; no hidden sampling or label access",
    }
    return selected, diagnostics


fit_sparse_features = fit_sparse
