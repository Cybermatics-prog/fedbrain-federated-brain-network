# SPDX-License-Identifier: MIT
"""Train-only Eq.11 numerical/convergence probe; never reads held-out results.

Use the existing full-fedbrain-s42 train IDs, saved train-only mean/std and
validated FC caches. No network access, model fitting, prediction or held-out
metric evaluation is performed. Artifacts contain only sparse diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Cap native CPU pools before importing NumPy, independent of parent settings.
for _variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '4'

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fedbrain.data import N_FEATURES, SITE_ORDER, load_subjects
from fedbrain.sparse import SparseFeatureSelector, _GroupedLogisticProblem


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def jaccard(first, second):
    first, second = set(map(int, first)), set(map(int, second))
    return len(first & second) / len(first | second) if first | second else 1.


def top_mask(coef, maximum=512):
    selected = np.flatnonzero(coef != 0)
    order = np.lexsort((selected, -np.abs(coef[selected])))
    return np.sort(selected[order[:maximum]])


def reference_gradient_loss(X, y, encoded, weights, biases):
    """Original implementation only for timing and one-step equivalence."""
    scores = np.einsum('ij,ij->i', X, weights[encoded]) + biases[encoded]
    probabilities = np.empty_like(scores)
    positive = scores >= 0
    probabilities[positive] = 1 / (1 + np.exp(-scores[positive]))
    exponential = np.exp(scores[~positive])
    probabilities[~positive] = exponential / (1 + exponential)
    residual = probabilities - y
    gradient = np.zeros_like(weights)
    np.add.at(gradient, encoded, X * residual[:, None])
    bias_gradient = np.bincount(encoded, weights=residual, minlength=len(weights))
    loss = float(np.logaddexp(0, scores).sum() - np.dot(y, scores))
    return gradient, bias_gradient, loss


def benchmark(X, y, segments):
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    group_ids, encoded = np.unique(segments, return_inverse=True)
    problem = _GroupedLogisticProblem(X, y, encoded, len(group_ids))
    rng = np.random.default_rng(42)
    weights = rng.normal(scale=.01, size=(len(group_ids), X.shape[1]))
    biases = rng.normal(scale=.01, size=len(group_ids))
    old_times, new_times = [], []
    for _ in range(5):
        start = time.perf_counter()
        old_grad, old_bias, old_loss = reference_gradient_loss(X, y, encoded, weights, biases)
        old_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        new_grad, new_bias = problem.gradient(weights, biases)
        new_loss = problem.loss(weights, biases)
        new_times.append(time.perf_counter() - start)
    errors = {'gradient_max_abs': float(np.max(np.abs(old_grad - new_grad))),
              'bias_gradient_max_abs': float(np.max(np.abs(old_bias - new_bias))),
              'objective_abs': float(abs(old_loss - new_loss))}
    if max(errors.values()) > 1e-10:
        raise AssertionError(f'Grouped BLAS equivalence gate failed: {errors}')
    return {'repetitions': 5, 'old_gradient_plus_loss_seconds': old_times,
            'grouped_gradient_plus_loss_seconds': new_times,
            'median_speedup': float(np.median(old_times) / np.median(new_times)),
            'absolute_errors': errors, 'equivalence_tolerance': 1e-10}


def cached_training_data(data_dir, source_run, samples):
    # Deliberately read only split.json and preprocessing.npz, never result.json,
    # predictions.json or checkpoint metrics. Only the TRAIN list is consumed.
    split_path = source_run / 'split.json'
    preprocessing_path = source_run / 'preprocessing.npz'
    training = json.loads(split_path.read_text(encoding='utf-8'))['train']
    manifest_subjects = {str(s['id']): s for s in load_subjects(data_dir)}
    with np.load(preprocessing_path, allow_pickle=False) as preprocessing:
        mean, std = preprocessing['mean'], preprocessing['std']
    if mean.shape != (N_FEATURES,) or std.shape != mean.shape or np.any(std <= 0):
        raise ValueError('Expected full-dimension saved training moments')
    clients = {site: {'X': [], 'y': [], 'segments': [], 'subjects': []} for site in SITE_ORDER}
    for row in training:
        subject = manifest_subjects[str(row['id'])]
        if row['site'] != subject['site'] or row['label'] != subject['label']:
            raise ValueError('Saved train subject metadata differs from public manifest')
        cache = data_dir / 'features' / f"{subject['id']}_w32_s1.npy"
        metadata = json.loads(cache.with_suffix('.json').read_text(encoding='utf-8'))
        source = Path(subject['path'])
        if metadata.get('source_sha256') != sha256(source) or metadata.get('window') != 32 or metadata.get('stride') != 1 or metadata.get('version') != 1:
            raise ValueError(f'Stale FC cache for training subject {subject["id"]}')
        array = np.load(cache, mmap_mode='r', allow_pickle=False)
        if array.ndim != 2 or array.shape[1] != N_FEATURES:
            raise ValueError('Invalid training FC cache shape')
        indices = np.linspace(0, len(array) - 1, min(len(array), samples), dtype=int)
        selected = np.asarray(array[indices], dtype=np.float32)
        if not np.isfinite(selected).all():
            raise ValueError('Non-finite training features')
        client = clients[subject['site']]
        client['X'].append(selected)
        client['y'].extend([subject['label']] * len(indices))
        client['segments'].extend(np.arange(len(indices)).tolist())
        client['subjects'].append(str(subject['id']))
    for client in clients.values():
        # Reproduce original float32 train standardization before solver float64.
        client['X'] = (np.concatenate(client['X']) - mean) / std
        client['y'] = np.asarray(client['y'], dtype=np.int64)
        client['segments'] = np.asarray(client['segments'], dtype=np.int64)
    provenance = {'split_path': str(split_path), 'split_sha256': sha256(split_path),
                  'preprocessing_path': str(preprocessing_path), 'preprocessing_sha256': sha256(preprocessing_path),
                  'train_subject_count': len(training),
                  'train_ids_sha256': hashlib.sha256('\n'.join(sorted(str(r['id']) for r in training)).encode()).hexdigest(),
                  'heldout_metrics_accessed': False, 'heldout_features_accessed': False,
                  'feature_source': 'existing SHA256-validated Fisher-z cache; training subjects only'}
    return clients, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=ROOT / 'runs' / 'full-fedbrain-s42')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'abide')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts' / 'sparse_probe.json')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        raise ValueError('Probe CPU threads must be1..4')
    print(json.dumps({'stage': 'preparing_train_only', 'threads': args.threads}), flush=True)
    clients, provenance = cached_training_data(args.data_dir.resolve(), args.source_run.resolve(), samples=8)
    report = {
        'tag': 'sparse-equivalent-grouped-blas-train-only-s42', 'seed': 42,
        'provenance': provenance,
        'configuration': {'l1': .01, 'coupling': 1., 'learning_rate': .1, 'steps': [40, 400],
                          'segments': 'eight evenly spaced relative-time bins', 'max_features': None,
                          'cap512': 'reported separately, never changes optimizer', 'cpu_threads': args.threads},
        'preregistered_gates': {'reference_absolute_tolerance': 1e-10,
                               'longer_objective_not_larger_tolerance': 1e-8,
                               'no_heldout_evaluation': True,
                               'convergence': 'original max-parameter-change<=1e-6, not a certified optimality gap'},
        'sites': {}, 'status': 'running',
    }
    union_scores = {40: np.zeros(N_FEATURES), 400: np.zeros(N_FEATURES)}
    with threadpool_limits(limits=args.threads):
        report['benchmark_NYU'] = benchmark(clients['NYU']['X'], clients['NYU']['y'], clients['NYU']['segments'])
        print(json.dumps({'stage': 'benchmark', **report['benchmark_NYU']}), flush=True)
        for site in SITE_ORDER:
            client, results, selectors = clients[site], {}, {}
            for steps in (40, 400):
                started = time.perf_counter()
                fitted = SparseFeatureSelector(steps=steps).fit(client['X'], client['y'], client['segments'])
                elapsed = time.perf_counter() - started
                diagnostics = fitted.diagnostics_
                results[str(steps)] = {'elapsed_seconds': elapsed, 'diagnostics': diagnostics,
                                       'shared_exact_zero_count': int(np.count_nonzero(fitted.coef_ == 0)),
                                       'selected_indices': fitted.selected_indices_.tolist(),
                                       'top512_indices': top_mask(fitted.coef_).tolist()}
                selectors[steps] = fitted
                union_scores[steps] = np.maximum(union_scores[steps], np.abs(fitted.coef_))
                print(json.dumps({'stage': 'site_fit', 'site': site, 'steps': steps, 'seconds': elapsed,
                                  'objective': diagnostics['objective_final'], 'nonzero': diagnostics['uncapped_nonzero'],
                                  'converged': diagnostics['converged'],
                                  'relative_last_drop': diagnostics['relative_objective_change_last']}), flush=True)
            short, long = selectors[40], selectors[400]
            if long.diagnostics_['objective_final'] > short.diagnostics_['objective_final'] + 1e-8:
                raise AssertionError(f'Longer optimization increased objective for {site}')
            comparison = {
                'objective_relative_reduction_40_to_400': (short.diagnostics_['objective_final'] - long.diagnostics_['objective_final']) / max(abs(short.diagnostics_['objective_final']), 1e-12),
                'uncapped_support_jaccard': jaccard(short.selected_indices_, long.selected_indices_),
                'top512_support_jaccard': jaccard(top_mask(short.coef_), top_mask(long.coef_)),
                'shared_nonzero_change': len(long.selected_indices_) - len(short.selected_indices_),
                'monotonic_400_within_1e-8': bool(np.all(np.diff(long.diagnostics_['objective_history']) <= 1e-8)),
            }
            report['sites'][site] = {'training_subjects': len(client['subjects']), 'training_windows': len(client['X']),
                                     'fits': results, 'comparison': comparison}
            save_json(args.output, report)
    report['union'] = {
        'nonzero_40': int(np.count_nonzero(union_scores[40])),
        'nonzero_400': int(np.count_nonzero(union_scores[400])),
        'uncapped_support_jaccard': jaccard(np.flatnonzero(union_scores[40]), np.flatnonzero(union_scores[400])),
        'top512_support_jaccard': jaccard(top_mask(union_scores[40]), top_mask(union_scores[400])),
        'top512_indices_40': top_mask(union_scores[40]).tolist(),
        'top512_indices_400': top_mask(union_scores[400]).tolist(),
    }
    report['summary'] = {'all_400_converged': all(site['fits']['400']['diagnostics']['converged'] for site in report['sites'].values()),
                         'all_long_objectives_not_larger': True,
                         'converged_sites_40': sum(site['fits']['40']['diagnostics']['converged'] for site in report['sites'].values()),
                         'converged_sites_400': sum(site['fits']['400']['diagnostics']['converged'] for site in report['sites'].values()),
                         'interpretation': 'Objective descent/mask stability only; no accuracy or generalization conclusion'}
    report['status'] = 'completed'
    save_json(args.output, report)
    print(json.dumps({'stage': 'completed', 'output': str(args.output), 'union': report['union'], 'summary': report['summary']}), flush=True)


if __name__ == '__main__':
    main()
